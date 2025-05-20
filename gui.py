import sys
import os
import json
import requests
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                           QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, 
                           QTabWidget, QRadioButton, QCheckBox, QComboBox, 
                           QProgressBar, QMessageBox, QLineEdit, QGroupBox,
                           QButtonGroup, QDialog, QTableWidget, QTableWidgetItem,
                           QHeaderView, QMenu, QMenuBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QPalette, QColor, QAction
from pathlib import Path
from typing import Optional, List
import time

# 导入处理模块
from utils import extract_audio, transcribe_audio, translate_subtitle_file, embed_subtitles, create_subtitle_file, temp_manager
from model_utils import check_environment, setup_environment, get_whisper_models, download_whisper_model

class EnvironmentDialog(QDialog):
    """环境配置对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("环境配置")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # 创建状态表格
        self.status_table = QTableWidget()
        self.status_table.setColumnCount(4)
        self.status_table.setHorizontalHeaderLabels(["组件", "状态", "版本", "详情"])
        self.status_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        # 刷新状态按钮
        refresh_btn = QPushButton("刷新状态")
        refresh_btn.clicked.connect(self.refresh_status)
        
        # 一键配置按钮
        setup_btn = QPushButton("一键配置环境")
        setup_btn.clicked.connect(self.setup_environment)
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(setup_btn)
        btn_layout.addStretch()
        
        layout.addWidget(self.status_table)
        layout.addLayout(btn_layout)
        
        # 初始化状态
        self.refresh_status()
        
    def refresh_status(self):
        """刷新环境状态"""
        status_list = check_environment()
        self.status_table.setRowCount(len(status_list))
        
        for i, status in enumerate(status_list):
            self.status_table.setItem(i, 0, QTableWidgetItem(status["name"]))
            self.status_table.setItem(i, 1, QTableWidgetItem(status["status"]))
            self.status_table.setItem(i, 2, QTableWidgetItem(status["version"]))
            self.status_table.setItem(i, 3, QTableWidgetItem(status["details"]))
    
    def setup_environment(self):
        """一键配置环境"""
        reply = QMessageBox.question(
            self,
            "确认",
            "这将安装所有必要的包和依赖，可能需要几分钟时间。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success, logs = setup_environment()
            
            # 显示安装日志
            log_text = "\n".join(logs)
            QMessageBox.information(
                self,
                "安装完成",
                f"环境配置{'成功' if success else '失败'}:\n\n{log_text}"
            )
            
            # 刷新状态
            self.refresh_status()

class ModelManagerDialog(QDialog):
    """模型管理对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Whisper模型管理")
        self.setMinimumSize(500, 300)
        
        layout = QVBoxLayout(self)
        
        # 创建模型表格
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["模型", "状态", "大小", "路径"])
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)  # 整行选择
        self.model_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)  # 单行选择
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # 禁止编辑
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_models)
        
        download_btn = QPushButton("下载选中模型")
        download_btn.clicked.connect(self.download_model)
        
        btn_layout.addWidget(refresh_btn)
        btn_layout.addWidget(download_btn)
        btn_layout.addStretch()
        
        layout.addWidget(self.model_table)
        layout.addLayout(btn_layout)
        
        # 初始化模型列表
        self.refresh_models()
        
    def refresh_models(self):
        """刷新模型列表"""
        # 获取可用的模型列表
        models = ["tiny", "base", "small", "medium", "large"]
        # 获取已下载的模型信息
        downloaded_models = get_whisper_models()
        downloaded_info = {model["name"]: model for model in downloaded_models}
        
        # 更新表格
        self.model_table.setRowCount(len(models))
        
        for i, model_name in enumerate(models):
            # 模型名称
            self.model_table.setItem(i, 0, QTableWidgetItem(model_name))
            
            # 获取下载信息
            if model_name in downloaded_info:
                info = downloaded_info[model_name]
                self.model_table.setItem(i, 1, QTableWidgetItem(info["status"]))
                self.model_table.setItem(i, 2, QTableWidgetItem(info["size"]))
                self.model_table.setItem(i, 3, QTableWidgetItem(info["path"]))
            else:
                self.model_table.setItem(i, 1, QTableWidgetItem("未下载"))
                self.model_table.setItem(i, 2, QTableWidgetItem("-"))
                self.model_table.setItem(i, 3, QTableWidgetItem("-"))
    
    def download_model(self):
        """下载选中的模型"""
        current_row = self.model_table.currentRow()
        if current_row >= 0:
            model_name = self.model_table.item(current_row, 0).text()
            
            reply = QMessageBox.question(
                self,
                "确认下载",
                f"确定要下载 {model_name} 模型吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                success, msg = download_whisper_model(model_name)
                QMessageBox.information(self, "下载结果", msg)
                if success:
                    self.refresh_models()
        else:
            QMessageBox.warning(self, "提示", "请先选择要下载的模型")

class OllamaModelFetcher(QThread):
    """Ollama模型获取线程"""
    models_fetched = pyqtSignal(list)
    fetch_error = pyqtSignal(str)
    
    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url
        
    def run(self):
        try:
            # 访问Ollama API获取模型列表
            response = requests.get(f"{self.server_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                self.models_fetched.emit(model_names)
            else:
                self.fetch_error.emit(f"获取模型列表失败: HTTP {response.status_code}")
        except Exception as e:
            self.fetch_error.emit(f"连接Ollama服务器失败: {str(e)}")

class VideoProcessThread(QThread):
    """视频处理线程"""
    progress_updated = pyqtSignal(int, str)
    process_finished = pyqtSignal(bool, str)
    
    def __init__(self, input_file: str, settings: dict):
        super().__init__()
        self.input_file = input_file
        self.settings = settings
        self.is_cancelled = False
        
    def run(self):
        try:
            # 设置输出路径
            if not self.settings['output_path']:
                base_name = os.path.splitext(self.input_file)[0]
                self.settings['output_path'] = f"{base_name}_subtitled.{self.settings['format']}"
                
            if not self.settings['srt_path']:
                self.settings['srt_path'] = os.path.splitext(self.input_file)[0] + "_original.srt"
                
            if not self.settings['audio_path']:
                self.settings['audio_path'] = os.path.splitext(self.input_file)[0] + ".wav"
            
            # 1. 提取音频
            self.progress_updated.emit(0, "正在提取音频...")
            audio_path = extract_audio(self.input_file)
            if not audio_path or self.is_cancelled:
                raise Exception("音频提取失败")
            
            # 2. 转录字幕
            self.progress_updated.emit(25, "正在转录字幕...")
            segments, full_text = transcribe_audio(audio_path)
            if not segments or self.is_cancelled:
                raise Exception("语音识别失败")
            
            # 创建字幕文件
            if not create_subtitle_file(segments, self.settings['srt_path']):
                raise Exception("字幕文件创建失败")
            
            # 3. 翻译字幕
            self.progress_updated.emit(50, "正在翻译字幕...")
            translated_srt_path = translate_subtitle_file(
                self.settings['srt_path'],
                target_lang=self.settings['target_lang'],
                use_ollama=self.settings['use_ollama'],
                model_name=self.settings['model_name'],
                retry_translate=self.settings['retry_translate'],
                base_url=self.settings['ollama_url']
            )
            
            if not translated_srt_path or self.is_cancelled:
                raise Exception("字幕翻译失败")
            
            # 4. 嵌入字幕
            self.progress_updated.emit(75, "正在嵌入字幕...")
            options = {
                'hard_sub': self.settings['hard_sub'],
                'format': self.settings['format'],
                'subtitle_style': self.settings['subtitle_style']
            }
            
            if not embed_subtitles(self.input_file, translated_srt_path, 
                                 self.settings['output_path'], options):
                raise Exception("字幕嵌入失败")
            
            self.progress_updated.emit(100, "处理完成！")
            self.process_finished.emit(True, "处理成功完成！")
            
        except Exception as e:
            self.process_finished.emit(False, f"处理失败: {str(e)}")
        finally:
            # 清理临时文件
            if not self.settings['keep_temp']:
                temp_manager.cleanup()
    
    def cancel(self):
        self.is_cancelled = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("视频字幕处理工具")
        self.setMinimumSize(800, 600)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 设置应用程序暗色主题
        self.set_dark_theme()
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 添加文件选择区域
        self.create_file_selection_area(layout)
        
        # 添加选项卡
        self.create_tabs(layout)
        
        # 添加进度区域
        self.create_progress_area(layout)
        
        # 设置样式
        self.apply_styles()
        
        # 初始化处理线程
        self.process_thread = None
        self.model_fetcher = None
        
        # 初始化Ollama模型列表
        self.refresh_ollama_models()
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        open_action = QAction("打开视频", self)
        open_action.triggered.connect(self.select_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具")
        
        env_action = QAction("环境配置", self)
        env_action.triggered.connect(self.show_environment_dialog)
        tools_menu.addAction(env_action)
        
        model_action = QAction("模型管理", self)
        model_action.triggered.connect(self.show_model_manager)
        tools_menu.addAction(model_action)
        
    def show_environment_dialog(self):
        """显示环境配置对话框"""
        dialog = EnvironmentDialog(self)
        dialog.exec()
        
    def show_model_manager(self):
        """显示模型管理对话框"""
        dialog = ModelManagerDialog(self)
        dialog.exec()

    def set_dark_theme(self):
        """设置暗色主题"""
        app = QApplication.instance()
        # 设置暗色调色板
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        app.setPalette(palette)
        
    def create_file_selection_area(self, parent_layout):
        """创建文件选择区域"""
        group = QGroupBox("文件选择")
        layout = QHBoxLayout(group)
        
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("拖放视频文件到此处或点击选择按钮")
        self.file_path.setReadOnly(True)
        
        select_btn = QPushButton("选择文件")
        select_btn.clicked.connect(self.select_file)
        select_btn.setFixedWidth(100)
        
        layout.addWidget(self.file_path)
        layout.addWidget(select_btn)
        
        parent_layout.addWidget(group)
        
    def create_tabs(self, parent_layout):
        """创建选项卡"""
        self.tabs = QTabWidget()
        
        # 基本设置选项卡
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # Whisper模型选择组
        whisper_group = QGroupBox("Whisper模型选择")
        whisper_layout = QHBoxLayout(whisper_group)
        
        self.whisper_models = QButtonGroup()
        whisper_model_names = ["tiny", "base", "small", "medium", "large"]
        for i, model in enumerate(whisper_model_names):
            radio = QRadioButton(model)
            if model == "medium":  # 默认选择medium
                radio.setChecked(True)
            self.whisper_models.addButton(radio, i)
            whisper_layout.addWidget(radio)
        
        basic_layout.addWidget(whisper_group)
        
        # 输出设置组
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout(output_group)
        
        self.output_type_srt = QRadioButton("生成单独的字幕文件")
        self.output_type_video = QRadioButton("生成带字幕的视频")
        self.output_type_both = QRadioButton("两者都生成")
        self.output_type_both.setChecked(True)
        
        output_layout.addWidget(self.output_type_srt)
        output_layout.addWidget(self.output_type_video)
        output_layout.addWidget(self.output_type_both)
        
        basic_layout.addWidget(output_group)
        
        # 字幕设置选项卡
        subtitle_tab = QWidget()
        subtitle_layout = QVBoxLayout(subtitle_tab)
        
        # 翻译设置组
        translation_group = QGroupBox("翻译设置")
        translation_layout = QVBoxLayout(translation_group)
        
        # 目标语言选择
        lang_layout = QHBoxLayout()
        lang_label = QLabel("目标语言:")
        self.target_lang = QComboBox()
        self.target_lang.addItems(["中文(简体)", "英语", "日语", "韩语"])
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.target_lang)
        lang_layout.addStretch()
        
        # 创建重试选项（提前定义）
        self.retry_translate = QCheckBox("翻译失败时自动重试")
        
        translation_layout.addLayout(lang_layout)
        translation_layout.addWidget(self.retry_translate)
        
        subtitle_layout.addWidget(translation_group)
        
        # 字幕样式组
        style_group = QGroupBox("字幕样式")
        style_layout = QVBoxLayout(style_group)
        
        self.sub_type_soft = QRadioButton("软字幕")
        self.sub_type_hard = QRadioButton("硬字幕")
        self.sub_type_soft.setChecked(True)
        
        style_layout.addWidget(self.sub_type_soft)
        style_layout.addWidget(self.sub_type_hard)
        
        style_select_layout = QHBoxLayout()
        style_label = QLabel("样式选择:")
        self.subtitle_style = QComboBox()
        self.subtitle_style.addItems(["默认", "描边", "阴影", "方框"])
        style_select_layout.addWidget(style_label)
        style_select_layout.addWidget(self.subtitle_style)
        style_select_layout.addStretch()
        
        style_layout.addLayout(style_select_layout)
        
        subtitle_layout.addWidget(style_group)
        subtitle_layout.addStretch()

        # 新增Ollama设置选项卡
        ollama_tab = QWidget()
        ollama_layout = QVBoxLayout(ollama_tab)
        
        # Ollama设置组
        ollama_group = QGroupBox("Ollama翻译设置")
        ollama_inner_layout = QVBoxLayout(ollama_group)
        
        # 启用Ollama
        self.use_ollama = QCheckBox("使用Ollama翻译")
        self.use_ollama.setChecked(True)
        
        # Ollama服务器地址和刷新按钮
        server_layout = QHBoxLayout()
        server_label = QLabel("服务器地址:")
        self.ollama_url = QLineEdit("http://localhost:11434")
        refresh_btn = QPushButton("刷新模型列表")
        refresh_btn.clicked.connect(self.refresh_ollama_models)
        
        server_layout.addWidget(server_label)
        server_layout.addWidget(self.ollama_url)
        server_layout.addWidget(refresh_btn)
        
        # Ollama模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        
        self.model_name = QComboBox()
        model_layout.addWidget(self.model_name)
        
        # 添加到Ollama设置组
        ollama_inner_layout.addWidget(self.use_ollama)
        ollama_inner_layout.addSpacing(10)  # 添加一些垂直间距
        ollama_inner_layout.addLayout(server_layout)
        ollama_inner_layout.addSpacing(10)  # 添加一些垂直间距
        ollama_inner_layout.addWidget(model_group)
        
        # 添加到Ollama标签页
        ollama_layout.addWidget(ollama_group)
        ollama_layout.addStretch()
        
        # 高级设置选项卡
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # 输出格式
        format_layout = QHBoxLayout()
        format_label = QLabel("输出格式:")
        self.output_format = QComboBox()
        self.output_format.addItems(["MP4", "MKV"])
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.output_format)
        format_layout.addStretch()
        
        # 其他选项
        self.keep_audio = QCheckBox("保存提取的音频文件")
        self.keep_srt = QCheckBox("保存原始字幕文件")
        self.keep_temp = QCheckBox("保留临时文件")
        self.show_log = QCheckBox("显示详细日志")
        
        advanced_layout.addLayout(format_layout)
        advanced_layout.addWidget(self.keep_audio)
        advanced_layout.addWidget(self.keep_srt)
        advanced_layout.addWidget(self.keep_temp)
        advanced_layout.addWidget(self.show_log)
        advanced_layout.addStretch()
        
        # 添加所有选项卡
        self.tabs.addTab(basic_tab, "基本设置")
        self.tabs.addTab(subtitle_tab, "字幕设置")
        self.tabs.addTab(ollama_tab, "翻译服务")
        self.tabs.addTab(advanced_tab, "高级设置")
        
        parent_layout.addWidget(self.tabs)
        
    def create_progress_area(self, parent_layout):
        """创建进度显示区域"""
        group = QGroupBox("处理进度")
        layout = QVBoxLayout(group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addLayout(btn_layout)
        
        parent_layout.addWidget(group)
        
    def apply_styles(self):
        """应用界面样式"""
        # 自定义样式
        custom_style = """
            QMainWindow {
                background-color: #353535;
            }
            QWidget {
                background-color: #353535;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 6px;
                margin-top: 1em;
                padding-top: 0.5em;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                color: #ffffff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d47a1;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 5px 15px;
                min-width: 80px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a367c;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #888888;
            }
            QProgressBar {
                border: 2px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #0d47a1;
                border-radius: 2px;
            }
            QLineEdit {
                padding: 5px;
                border: 2px solid #555555;
                border-radius: 4px;
                background-color: #2a2a2a;
                color: white;
                min-height: 25px;
            }
            QComboBox {
                padding: 5px;
                border: 2px solid #555555;
                border-radius: 4px;
                background-color: #2a2a2a;
                color: white;
                min-height: 25px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
            QTabWidget::pane {
                border: 2px solid #555555;
                border-radius: 6px;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #353535;
                border: 2px solid #555555;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 5px 10px;
                margin-right: 2px;
                color: white;
            }
            QTabBar::tab:selected {
                background-color: #0d47a1;
            }
            QTabBar::tab:hover {
                background-color: #1565c0;
            }
            QCheckBox {
                spacing: 5px;
                color: white;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555555;
                border-radius: 3px;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #0d47a1;
            }
            QRadioButton {
                spacing: 5px;
                color: white;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555555;
                border-radius: 9px;
                background-color: #2a2a2a;
            }
            QRadioButton::indicator:checked {
                background-color: #0d47a1;
            }
            QLabel {
                color: white;
            }
        """
        self.setStyleSheet(custom_style)

    def select_file(self):
        """选择视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.mkv *.avi *.mov);;所有文件 (*.*)"
        )
        if file_name:
            self.file_path.setText(file_name)
            
    def dragEnterEvent(self, event: QDragEnterEvent):
        """处理拖入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        """处理放下事件"""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.file_path.setText(files[0])
            
    def refresh_ollama_models(self):
        """刷新Ollama模型列表"""
        try:
            # 保存当前选择的模型
            current_model = self.model_name.currentText()
            
            # 清空模型列表
            self.model_name.clear()
            
            # 创建并启动模型获取线程
            self.model_fetcher = OllamaModelFetcher(self.ollama_url.text().strip())
            self.model_fetcher.models_fetched.connect(self.update_model_list)
            self.model_fetcher.fetch_error.connect(self.show_model_fetch_error)
            self.model_fetcher.start()
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"刷新模型列表失败: {str(e)}")
    
    def update_model_list(self, models: List[str]):
        """更新模型列表"""
        self.model_name.clear()
        self.model_name.addItems(models)
        
        # 如果列表为空，添加默认模型
        if not models:
            self.model_name.addItem("huihui_ai/qwen3-abliterated")
    
    def show_model_fetch_error(self, error: str):
        """显示模型获取错误"""
        QMessageBox.warning(self, "错误", f"获取模型列表失败: {error}")
        # 添加默认模型
        self.model_name.addItem("huihui_ai/qwen3-abliterated")

    def get_settings(self) -> dict:
        """获取当前设置"""
        return {
            'output_path': '',  # 将在处理线程中设置
            'srt_path': '',     # 将在处理线程中设置
            'audio_path': '',   # 将在处理线程中设置
            'whisper_model': self.whisper_models.checkedButton().text(),
            'target_lang': 'zh-CN' if self.target_lang.currentText() == "中文(简体)" else 'en',
            'use_ollama': self.use_ollama.isChecked(),
            'ollama_url': self.ollama_url.text().strip(),
            'model_name': self.model_name.currentText(),
            'retry_translate': self.retry_translate.isChecked(),
            'hard_sub': self.sub_type_hard.isChecked(),
            'format': self.output_format.currentText().lower(),
            'subtitle_style': self.subtitle_style.currentText().lower(),
            'keep_temp': self.keep_temp.isChecked()
        }
        
    def start_processing(self):
        """开始处理"""
        if not self.file_path.text():
            QMessageBox.warning(self, "警告", "请先选择视频文件！")
            return
            
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动处理线程
        self.process_thread = VideoProcessThread(self.file_path.text(), self.get_settings())
        self.process_thread.progress_updated.connect(self.update_progress)
        self.process_thread.process_finished.connect(self.process_completed)
        self.process_thread.start()
        
    def cancel_processing(self):
        """取消处理"""
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.cancel()
            self.process_thread.wait()
            self.status_label.setText("处理已取消")
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            
    def update_progress(self, value: int, status: str):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(status)
        
    def process_completed(self, success: bool, message: str):
        """处理完成"""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "错误", message)
            
        self.status_label.setText("就绪")

def main():
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 