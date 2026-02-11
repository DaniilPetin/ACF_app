import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import math
from scipy import signal as scipy_signal
from scipy.signal import hilbert
from pathlib import Path
import json
import concurrent.futures
import time
import threading

class AutocorrelationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ автокорреляции импульсных последовательностей")
        self.root.geometry("1500x1000")
        
        # Параметры по умолчанию
        self.default_params = {
            'ricker_freq': 100.0,
            'duration': 10.0,
            'start_freq': 25.105,
            'end_freq': 48.0,
            'dt': 0.001,
            'max_lag': 500,
            'law_type': 'exponential',
            'variable_amplitude': False
        }
        
        # Коэффициенты для компенсационного закона (базовые)
        self.compensation_coefficients = [
            [0.0005, -0.0023, 0.0785, 1.1904, 25.019],
            [0.000035, -0.0003, 0.0160, 0.5952, 25.019],  
            [0.000026, -0.000045, 0.0049, 0.2976, 25.019],
            [0.000015, -0.000045, 0.0012, 0.1488, 25.019]
        ]
        
        # Текущие параметры
        self.params = self.default_params.copy()
        
        # Словари для хранения ручных границ палитры для каждого типа карты
        self.manual_vmin = {'area': None, 'center_freq': None, 'impulse_count': None, 'envelope_area': None}
        self.manual_vmax = {'area': None, 'center_freq': None, 'impulse_count': None, 'envelope_area': None}
        
        # Флаг для остановки расчета
        self.calculation_stopped = False
        self.calculation_thread = None
        
        # Настройка стилей для черного шрифта на кнопках
        self.setup_button_styles()
        
        # Создание интерфейса
        self.setup_ui()
        
        # Первоначальный расчет
        self.update_plots()
    
    def setup_button_styles(self):
        """Настройка стилей кнопок с черным шрифтом"""
        style = ttk.Style()
        
        # Стандартные кнопки
        style.configure('TButton', foreground='black', font=('Arial', 9))
        
        # Зеленая кнопка (Подбор параметров)
        style.configure('Green.TButton', 
                       foreground='black',
                       font=('Arial', 10, 'bold'))
        
        # Синяя кнопка (Выгрузить карту)
        style.configure('Blue.TButton', 
                       foreground='black',
                       font=('Arial', 9, 'bold'))
        
        # Кнопки с расширенными функциями
        style.configure('Optimization.TButton', 
                       foreground='black',
                       font=('Arial', 9))
    
    def setup_ui(self):
        """Создание основного интерфейса"""
        # Основное разделение окна
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Панель параметров слева
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Верхняя часть левой панели - параметры
        control_frame = ttk.LabelFrame(left_panel, text="Параметры", padding=(10, 5))
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Параметры импульса Рикера
        ttk.Label(control_frame, text="ИМПУЛЬС РИКЕРА", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W)
        
        ttk.Label(control_frame, text="Частота (Гц):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ricker_freq_var = tk.DoubleVar(value=self.params['ricker_freq'])
        self.ricker_freq_entry = ttk.Entry(control_frame, textvariable=self.ricker_freq_var, width=15)
        self.ricker_freq_entry.grid(row=1, column=1, padx=5, pady=2)
        
        # Параметры последовательности
        ttk.Label(control_frame, text="ПОСЛЕДОВАТЕЛЬНОСТЬ", font=('Arial', 10, 'bold')).grid(
            row=2, column=0, columnspan=2, pady=(10, 5), sticky=tk.W)
        
        ttk.Label(control_frame, text="Длительность (сек):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.duration_var = tk.DoubleVar(value=self.params['duration'])
        self.duration_entry = ttk.Entry(control_frame, textvariable=self.duration_var, width=15)
        self.duration_entry.grid(row=3, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Начальная частота (Гц):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.start_freq_var = tk.DoubleVar(value=self.params['start_freq'])
        self.start_freq_entry = ttk.Entry(control_frame, textvariable=self.start_freq_var, width=15)
        self.start_freq_entry.grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Конечная частота (Гц):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.end_freq_var = tk.DoubleVar(value=self.params['end_freq'])
        self.end_freq_entry = ttk.Entry(control_frame, textvariable=self.end_freq_var, width=15)
        self.end_freq_entry.grid(row=5, column=1, padx=5, pady=2)
        
        # Закон изменения частоты
        ttk.Label(control_frame, text="Закон изменения частоты:", 
                 font=('Arial', 9, 'bold')).grid(row=6, column=0, columnspan=2, pady=(10, 5), sticky=tk.W)
        
        self.law_type_var = tk.StringVar(value=self.params['law_type'])
        law_types = [('Линейный', 'linear'), 
                    ('Квадратичный', 'quadratic'), 
                    ('Экспоненциальный', 'exponential'),
                    ('Компенсационный', 'compensation'),
                    ('Гиперболический', 'hyperbolic')]
        
        for i, (text, value) in enumerate(law_types):
            rb = ttk.Radiobutton(control_frame, text=text, variable=self.law_type_var, 
                                value=value, command=self.on_law_type_change)
            rb.grid(row=7+i, column=0, columnspan=2, sticky=tk.W, padx=20, pady=2)
        
        # Новая опция: изменение амплитуды импульсов
        self.var_amp_var = tk.BooleanVar(value=self.params['variable_amplitude'])
        self.var_amp_check = ttk.Checkbutton(control_frame, text="Учет изменения амплитуды ударов",
                                           variable=self.var_amp_var, command=self.on_var_amp_change)
        self.var_amp_check.grid(row=12, column=0, columnspan=2, sticky=tk.W, padx=20, pady=(10, 5))
        
        # Кнопки управления
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=13, column=0, columnspan=2, pady=(20, 10))
        
        ttk.Button(button_frame, text="Подбор параметров", 
                  command=self.open_parameter_optimization, width=25,
                  style='Green.TButton').pack(pady=5)
        
        ttk.Button(button_frame, text="Перестроить графики", 
                  command=self.update_plots, width=25).pack(pady=5)
        
        ttk.Button(button_frame, text="Сохранить АКФ (.txt)", 
                  command=self.save_autocorrelation, width=25).pack(pady=5)
        
        ttk.Button(button_frame, text="Сохранить времена ударов (.txt)", 
                  command=self.save_impulse_times, width=25).pack(pady=5)
        
        ttk.Button(button_frame, text="Сохранить свертку (.txt)", 
                  command=self.save_convolution, width=25).pack(pady=5)
        
        ttk.Button(button_frame, text="Сохранить все параметры", 
                  command=self.save_all_parameters, width=25).pack(pady=5)
        
        ttk.Button(button_frame, text="По умолчанию", 
                  command=self.reset_to_defaults, width=25).pack(pady=5)
        
        # График роста частоты (увеличенный, фиксированный размер)
        freq_frame = ttk.Frame(left_panel)
        freq_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        freq_frame.pack_propagate(False)
        freq_frame.config(width=350, height=350)
        
        self.fig_freq = plt.Figure(figsize=(3.5, 3.5), dpi=100)
        self.ax_freq = self.fig_freq.add_subplot(111)
        self.canvas_freq = FigureCanvasTkAgg(self.fig_freq, freq_frame)
        self.canvas_freq.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Область графиков справа
        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Верхние графики
        self.fig_top = plt.Figure(figsize=(12, 4), dpi=100)
        self.ax_impulse = self.fig_top.add_subplot(121)
        self.ax_convolution = self.fig_top.add_subplot(122)
        self.canvas_top = FigureCanvasTkAgg(self.fig_top, graph_frame)
        self.canvas_top.get_tk_widget().pack(fill=tk.BOTH, expand=False, pady=(0, 5))
        
        # Нижние графики
        self.fig_bottom = plt.Figure(figsize=(12, 6), dpi=100)
        self.ax_autocorr = self.fig_bottom.add_subplot(121)
        self.ax_spectrum = self.fig_bottom.add_subplot(122)
        self.canvas_bottom = FigureCanvasTkAgg(self.fig_bottom, graph_frame)
        self.canvas_bottom.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(5, 0))
    
    def open_parameter_optimization(self):
        """Открыть окно подбора параметров"""
        if not self.get_parameters():
            return
        
        # Создаем новое окно
        self.optimization_window = tk.Toplevel(self.root)
        self.optimization_window.title("Подбор параметров частот")
        self.optimization_window.geometry("1200x800")
        
        # Сохраняем текущие параметры
        self.fixed_params = {
            'ricker_freq': self.params['ricker_freq'],
            'duration': self.params['duration'],
            'law_type': self.params['law_type'],
            'variable_amplitude': self.params['variable_amplitude'],
            'dt': self.params['dt'],
            'max_lag': self.params['max_lag']
        }
        
        # Параметры перебора по умолчанию
        self.opt_params = {
            'start_freq_min': 10.0,
            'start_freq_max': 25.0,
            'start_freq_step': 2.0,
            'end_freq_min': 30.0,
            'end_freq_max': 60.0,
            'end_freq_step': 2.0,
            'heatmap_type': 'envelope_area'
        }
        
        # Создаем интерфейс окна подбора
        self.setup_optimization_ui()
        
        # Первоначальный расчет
        self.calculate_heatmap()
    
    def setup_optimization_ui(self):
        """Создание интерфейса окна подбора параметров"""
        main_opt_frame = ttk.Frame(self.optimization_window, padding=10)
        main_opt_frame.pack(fill=tk.BOTH, expand=True)
        
        # Верхняя панель с параметрами перебора
        param_frame = ttk.LabelFrame(main_opt_frame, text="Параметры расчета", padding=(10, 5))
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # ФИКСИРОВАННЫЙ ПАРАМЕТР
        fixed_frame = ttk.Frame(param_frame)
        fixed_frame.grid(row=0, column=0, columnspan=6, sticky=tk.W, pady=(0, 10))
        
        ttk.Label(fixed_frame, text="ФИКСИРОВАННЫЙ ПАРАМЕТР:", 
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        fixed_text = f"Частота Рикера: {self.fixed_params['ricker_freq']} Гц"
        ttk.Label(fixed_frame, text=fixed_text, font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Длительность последовательности с кнопками-стрелками
        duration_frame = ttk.Frame(param_frame)
        duration_frame.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(0, 10))
        
        ttk.Label(duration_frame, text="Длительность (сек):", 
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.duration_left_btn = ttk.Button(duration_frame, text="←", width=3,
                                          command=lambda: self.change_duration(-5))
        self.duration_left_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.opt_duration_var = tk.DoubleVar(value=self.fixed_params['duration'])
        self.opt_duration_entry = ttk.Entry(duration_frame, textvariable=self.opt_duration_var, 
                                           width=8, justify='center')
        self.opt_duration_entry.pack(side=tk.LEFT, padx=5)
        self.opt_duration_entry.bind('<Return>', lambda e: self.on_duration_changed())
        
        self.duration_right_btn = ttk.Button(duration_frame, text="→", width=3,
                                           command=lambda: self.change_duration(5))
        self.duration_right_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Выбор типа последовательности
        law_frame = ttk.Frame(param_frame)
        law_frame.grid(row=1, column=4, columnspan=3, sticky=tk.W, pady=(0, 10), padx=(40, 0))
        
        ttk.Label(law_frame, text="Тип последовательности:", 
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.opt_law_type_var = tk.StringVar(value=self.fixed_params['law_type'])
        law_types = [('Линейный', 'linear'), 
                    ('Квадратичный', 'quadratic'), 
                    ('Экспоненциальный', 'exponential'),
                    ('Компенсационный', 'compensation'),
                    ('Гиперболический', 'hyperbolic')]
        
        law_combo = ttk.Combobox(law_frame, textvariable=self.opt_law_type_var, 
                                values=[lt[0] for lt in law_types], state='readonly', width=15)
        law_combo.pack(side=tk.LEFT, padx=5)
        
        law_value_map = {lt[0]: lt[1] for lt in law_types}
        law_combo.bind('<<ComboboxSelected>>', 
                      lambda e: self.on_opt_law_type_change(law_value_map[law_combo.get()]))
        
        # Изменение амплитуды
        amp_frame = ttk.Frame(param_frame)
        amp_frame.grid(row=1, column=7, sticky=tk.W, pady=(0, 10), padx=(20, 0))
        
        self.opt_var_amp_var = tk.BooleanVar(value=self.fixed_params['variable_amplitude'])
        self.opt_var_amp_check = ttk.Checkbutton(amp_frame, text="Изменение амплитуды",
                                               variable=self.opt_var_amp_var, 
                                               command=self.on_opt_var_amp_change)
        self.opt_var_amp_check.pack(side=tk.LEFT)
        
        # Тип тепловой карты
        heatmap_type_frame = ttk.Frame(param_frame)
        heatmap_type_frame.grid(row=2, column=0, columnspan=6, sticky=tk.W, pady=(10, 5))
        
        ttk.Label(heatmap_type_frame, text="Тип карты:", 
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.heatmap_type_var = tk.StringVar(value=self.opt_params['heatmap_type'])
        heatmap_types = [('Площадь АКФ', 'area'), 
                        ('Центральная частота', 'center_freq'),
                        ('Число импульсов', 'impulse_count'),
                        ('Пл. под огиб. АКФ', 'envelope_area')]
        
        for text, value in heatmap_types:
            rb = ttk.Radiobutton(heatmap_type_frame, text=text, variable=self.heatmap_type_var,
                               value=value, command=self.on_heatmap_type_changed)
            rb.pack(side=tk.LEFT, padx=5)
        
        # Параметры перебора частот - начальные частоты
        ttk.Label(param_frame, text="НАЧАЛЬНЫЕ ЧАСТОТЫ", 
                 font=('Arial', 9, 'bold')).grid(row=3, column=0, columnspan=6, sticky=tk.W, pady=(10, 5))
        
        ttk.Label(param_frame, text="Мин. (Гц):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.start_freq_min_var = tk.DoubleVar(value=self.opt_params['start_freq_min'])
        ttk.Entry(param_frame, textvariable=self.start_freq_min_var, width=8).grid(row=4, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Макс. (Гц):").grid(row=4, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.start_freq_max_var = tk.DoubleVar(value=self.opt_params['start_freq_max'])
        ttk.Entry(param_frame, textvariable=self.start_freq_max_var, width=8).grid(row=4, column=3, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Шаг (Гц):").grid(row=4, column=4, sticky=tk.W, pady=2, padx=(10, 0))
        self.start_freq_step_var = tk.DoubleVar(value=self.opt_params['start_freq_step'])
        ttk.Entry(param_frame, textvariable=self.start_freq_step_var, width=8).grid(row=4, column=5, padx=5, pady=2)
        
        # Параметры перебора частот - конечные частоты
        ttk.Label(param_frame, text="КОНЕЧНЫЕ ЧАСТОТЫ", 
                 font=('Arial', 9, 'bold')).grid(row=5, column=0, columnspan=6, sticky=tk.W, pady=(10, 5))
        
        ttk.Label(param_frame, text="Мин. (Гц):").grid(row=6, column=0, sticky=tk.W, pady=2)
        self.end_freq_min_var = tk.DoubleVar(value=self.opt_params['end_freq_min'])
        ttk.Entry(param_frame, textvariable=self.end_freq_min_var, width=8).grid(row=6, column=1, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Макс. (Гц):").grid(row=6, column=2, sticky=tk.W, pady=2, padx=(10, 0))
        self.end_freq_max_var = tk.DoubleVar(value=self.opt_params['end_freq_max'])
        ttk.Entry(param_frame, textvariable=self.end_freq_max_var, width=8).grid(row=6, column=3, padx=5, pady=2)
        
        ttk.Label(param_frame, text="Шаг (Гц):").grid(row=6, column=4, sticky=tk.W, pady=2, padx=(10, 0))
        self.end_freq_step_var = tk.DoubleVar(value=self.opt_params['end_freq_step'])
        ttk.Entry(param_frame, textvariable=self.end_freq_step_var, width=8).grid(row=6, column=5, padx=5, pady=2)
        
        # Кнопка расчета
        ttk.Button(param_frame, text="Пересчитать", 
                  command=self.calculate_heatmap, width=20).grid(row=6, column=6, padx=(20, 0), pady=5)
        
        # Кнопка остановки расчета
        self.stop_button = ttk.Button(param_frame, text="Остановить расчет", 
                                     command=self.stop_calculation, width=20, state='disabled')
        self.stop_button.grid(row=6, column=7, padx=(20, 0), pady=5)
        
        # Область с тепловой картой и палитрой
        heatmap_frame = ttk.Frame(main_opt_frame)
        heatmap_frame.pack(fill=tk.BOTH, expand=True)
        
        # Тепловая карта (фиксированный размер)
        heatmap_left = ttk.Frame(heatmap_frame)
        heatmap_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        heatmap_left.pack_propagate(False)
        heatmap_left.config(width=800, height=550)
        
        self.fig_heatmap = plt.Figure(figsize=(8, 5.5), dpi=100)
        self.ax_heatmap = self.fig_heatmap.add_subplot(111)
        self.canvas_heatmap = FigureCanvasTkAgg(self.fig_heatmap, heatmap_left)
        self.canvas_heatmap.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правая панель
        heatmap_right = ttk.Frame(heatmap_frame)
        heatmap_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        heatmap_right.pack_propagate(False)
        heatmap_right.config(width=200, height=550)
        
        # Фрейм для палитры
        colorbar_frame = ttk.Frame(heatmap_right)
        colorbar_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig_colorbar = plt.Figure(figsize=(2, 4), dpi=100)
        self.ax_colorbar = self.fig_colorbar.add_subplot(111)
        self.canvas_colorbar = FigureCanvasTkAgg(self.fig_colorbar, colorbar_frame)
        self.canvas_colorbar.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Фрейм для ручных границ палитры
        palette_control_frame = ttk.LabelFrame(heatmap_right, text="Границы палитры", padding=(5, 5))
        palette_control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 5))
        
        # Минимальное значение
        min_frame = ttk.Frame(palette_control_frame)
        min_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(min_frame, text="Мин:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_palette_var = tk.StringVar(value="авто")
        self.min_palette_entry = ttk.Entry(min_frame, textvariable=self.min_palette_var, width=10)
        self.min_palette_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Максимальное значение
        max_frame = ttk.Frame(palette_control_frame)
        max_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(max_frame, text="Макс:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_palette_var = tk.StringVar(value="авто")
        self.max_palette_entry = ttk.Entry(max_frame, textvariable=self.max_palette_var, width=10)
        self.max_palette_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # Фрейм для кнопок управления палитрой
        palette_buttons_frame = ttk.Frame(palette_control_frame)
        palette_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(palette_buttons_frame, text="Применить", 
                  command=self.apply_manual_palette_bounds, width=12).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(palette_buttons_frame, text="Авто", 
                  command=self.reset_palette_bounds, width=12).pack(side=tk.LEFT)
        
        # Новая кнопка для выгрузки тепловой карты в TXT
        export_button_frame = ttk.Frame(heatmap_right)
        export_button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(5, 10))
        
        self.export_button = ttk.Button(export_button_frame, text="Выгрузить карту (.txt)", 
                                       command=self.export_heatmap_data, width=25,
                                       style='Blue.TButton')
        self.export_button.pack(pady=10)
    
    def stop_calculation(self):
        """Остановка расчета тепловой карты"""
        self.calculation_stopped = True
        self.stop_button.config(state='disabled')
    
    def export_heatmap_data(self):
        """Выгрузка данных тепловой карты в TXT файл"""
        if not hasattr(self, 'current_heatmap_data'):
            messagebox.showwarning("Нет данных", "Сначала постройте тепловую карту")
            return
        
        start_freqs, end_freqs, matrix, heatmap_type = self.current_heatmap_data
        
        heatmap_metric_names = {
            'area': 'Площадь_под_АКФ',
            'center_freq': 'Центральная_частота_Гц',
            'impulse_count': 'Число_импульсов',
            'envelope_area': 'Площадь_под_огибающей_АКФ'
        }
        
        law_type_names = {
            'linear': 'линейный',
            'quadratic': 'квадратичный',
            'exponential': 'экспоненциальный',
            'compensation': 'компенсационный',
            'hyperbolic': 'гиперболический'
        }
        
        metric_name = heatmap_metric_names.get(heatmap_type, 'метрика')
        law_name = law_type_names.get(self.fixed_params['law_type'], 'последовательность')
        
        file_name = f"Карта_{metric_name}_{law_name}_{self.fixed_params['ricker_freq']:.0f}Гц_{self.fixed_params['duration']:.0f}сек.txt"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
            initialfile=file_name
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Начальная_частота_Гц\tКонечная_частота_Гц\tДлительность_сек\tЧастота_Рикера_Гц\t{metric_name}\n")
                
                for j in range(len(end_freqs)):
                    for i in range(len(start_freqs)):
                        start_freq = start_freqs[i]
                        end_freq = end_freqs[j]
                        value = matrix[j, i]
                        
                        if heatmap_type == 'area':
                            value_str = f"{value:.3f}"
                        elif heatmap_type == 'center_freq':
                            value_str = f"{value:.1f}"
                        elif heatmap_type == 'impulse_count':
                            value_str = f"{value:.0f}"
                        else:
                            value_str = f"{value:.4f}"
                        
                        f.write(f"{start_freq:.1f}\t{end_freq:.1f}\t{self.fixed_params['duration']:.1f}\t{self.fixed_params['ricker_freq']:.1f}\t{value_str}\n")
            
            messagebox.showinfo("Успех", f"Тепловая карта успешно экспортирована в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def change_duration(self, delta):
        """Изменение длительности с помощью стрелок"""
        try:
            current_duration = float(self.opt_duration_var.get())
            new_duration = current_duration + delta
            
            if new_duration < 1:
                new_duration = 1
            elif new_duration > 100:
                new_duration = 100
            
            self.opt_duration_var.set(new_duration)
            self.fixed_params['duration'] = new_duration
            self.calculate_heatmap()
            
        except ValueError:
            self.opt_duration_var.set(self.fixed_params['duration'])
    
    def on_heatmap_type_changed(self):
        """Обработка изменения типа карты"""
        current_type = self.heatmap_type_var.get()
        self.update_palette_fields_for_type(current_type)
        self.calculate_heatmap()
    
    def update_palette_fields_for_type(self, heatmap_type):
        """Обновление полей ввода палитры для указанного типа карты"""
        vmin = self.manual_vmin[heatmap_type]
        vmax = self.manual_vmax[heatmap_type]
        
        if vmin is None:
            self.min_palette_var.set("авто")
        else:
            if heatmap_type == 'area':
                self.min_palette_var.set(f"{vmin:.3f}")
            elif heatmap_type == 'center_freq':
                self.min_palette_var.set(f"{vmin:.1f}")
            elif heatmap_type == 'impulse_count':
                self.min_palette_var.set(f"{vmin:.0f}")
            else:
                self.min_palette_var.set(f"{vmin:.4f}")
        
        if vmax is None:
            self.max_palette_var.set("авто")
        else:
            if heatmap_type == 'area':
                self.max_palette_var.set(f"{vmax:.3f}")
            elif heatmap_type == 'center_freq':
                self.max_palette_var.set(f"{vmax:.1f}")
            elif heatmap_type == 'impulse_count':
                self.max_palette_var.set(f"{vmax:.0f}")
            else:
                self.max_palette_var.set(f"{vmax:.4f}")
    
    def apply_manual_palette_bounds(self):
        """Применить ручные границы палитры"""
        try:
            heatmap_type = self.heatmap_type_var.get()
            min_val_str = self.min_palette_var.get().strip()
            max_val_str = self.max_palette_var.get().strip()
            
            if min_val_str.lower() == "авто" or min_val_str == "":
                self.manual_vmin[heatmap_type] = None
            else:
                self.manual_vmin[heatmap_type] = float(min_val_str)
            
            if max_val_str.lower() == "авто" or max_val_str == "":
                self.manual_vmax[heatmap_type] = None
            else:
                self.manual_vmax[heatmap_type] = float(max_val_str)
            
            self.update_heatmap_visualization()
            
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное значение границ палитры")
            self.manual_vmin[heatmap_type] = None
            self.manual_vmax[heatmap_type] = None
            self.update_palette_fields_for_type(heatmap_type)
    
    def reset_palette_bounds(self):
        """Сбросить границы палитры к автоматическим"""
        heatmap_type = self.heatmap_type_var.get()
        self.manual_vmin[heatmap_type] = None
        self.manual_vmax[heatmap_type] = None
        self.update_palette_fields_for_type(heatmap_type)
        self.update_heatmap_visualization()
    
    def update_heatmap_visualization(self):
        """Обновить визуализацию тепловой карты с текущими данными"""
        if hasattr(self, 'current_heatmap_data'):
            start_freqs, end_freqs, matrix, heatmap_type = self.current_heatmap_data
            self.update_heatmap(start_freqs, end_freqs, matrix, heatmap_type)
    
    def on_duration_changed(self):
        """Обработка изменения длительности из поля ввода"""
        try:
            new_duration = float(self.opt_duration_var.get())
            if new_duration < 1:
                raise ValueError("Длительность должна быть >= 1")
            if new_duration > 100:
                raise ValueError("Длительность не должна превышать 100 сек")
            
            self.fixed_params['duration'] = new_duration
            self.calculate_heatmap()
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректное значение длительности: {str(e)}")
            self.opt_duration_var.set(self.fixed_params['duration'])
    
    def on_opt_law_type_change(self, law_type):
        """Обработка изменения типа последовательности в окне оптимизации"""
        self.fixed_params['law_type'] = law_type
        self.calculate_heatmap()
    
    def on_opt_var_amp_change(self):
        """Обработка изменения опции переменной амплитуды в окне оптимизации"""
        self.fixed_params['variable_amplitude'] = self.opt_var_amp_var.get()
        self.calculate_heatmap()
    
    def calculate_single_point(self, task_data):
        """Функция для расчета одной точки тепловой карты (выполняется в потоке)"""
        try:
            i = task_data['i']
            j = task_data['j']
            start_freq = task_data['start_freq']
            end_freq = task_data['end_freq']
            fixed_params = task_data['fixed_params']
            heatmap_type = task_data['heatmap_type']
            
            temp_params = fixed_params.copy()
            temp_params['start_freq'] = start_freq
            temp_params['end_freq'] = end_freq
            
            duration = temp_params['duration']
            dt = temp_params['dt']
            total_samples = int(duration / dt)
            
            MAX_SAMPLES = 1000000
            if total_samples > MAX_SAMPLES:
                temp_params['dt'] = max(dt, duration / MAX_SAMPLES)
                dt = temp_params['dt']
            
            impulse_times = self.create_impulse_times_only(temp_params)
            
            if len(impulse_times) > 100000:
                return (i, j, 0)
            
            signal_len = int(duration / dt) + 1
            if signal_len > 1000000:
                return (i, j, 0)
                
            signal = np.zeros(signal_len)
            if temp_params['variable_amplitude'] and len(impulse_times) > 0:
                for idx, t in enumerate(impulse_times):
                    sample_idx = int(t / dt)
                    if 0 <= sample_idx < len(signal):
                        amplitude = 1.0 + (idx / (len(impulse_times) - 1)) if len(impulse_times) > 1 else 1.0
                        signal[sample_idx] = amplitude
            else:
                for t in impulse_times:
                    sample_idx = int(t / dt)
                    if 0 <= sample_idx < len(signal):
                        signal[sample_idx] = 1.0
            
            wavelet = self.ricker_wavelet_with_params(temp_params['ricker_freq'], temp_params)
            
            if len(signal) + len(wavelet) - 1 > 2000000:
                return (i, j, 0)
            
            convolution = np.convolve(signal, wavelet, mode='same')
            
            max_lag = temp_params['max_lag']
            n = len(convolution)
            
            if n > 200000:
                max_lag = min(max_lag, 200)
            
            autocorr = np.correlate(convolution, convolution, mode='full')
            autocorr = autocorr[n-1-max_lag:n+max_lag]
            
            max_val = np.max(np.abs(autocorr))
            if max_val > 0:
                autocorr = autocorr / max_val
            
            if heatmap_type == 'area':
                lag_times = np.arange(-max_lag, max_lag + 1) * dt
                area = np.sum(np.abs(autocorr)) * (lag_times[1] - lag_times[0])
                value = area
                
            elif heatmap_type == 'center_freq':
                center_freq = (start_freq + end_freq) / 2
                value = center_freq
                
            elif heatmap_type == 'impulse_count':
                num_impulses = len(impulse_times)
                value = num_impulses
                
            else:
                model_wavelet = wavelet
                
                if np.max(np.abs(model_wavelet)) > 0:
                    model_wavelet = model_wavelet / np.max(np.abs(model_wavelet))
                
                n_wavelet = len(model_wavelet)
                n_autocorr = len(autocorr)
                
                if n_wavelet > n_autocorr:
                    model_scaled = model_wavelet[(n_wavelet - n_autocorr) // 2: (n_wavelet - n_autocorr) // 2 + n_autocorr]
                else:
                    model_scaled = np.zeros(n_autocorr)
                    start_idx = (n_autocorr - n_wavelet) // 2
                    model_scaled[start_idx:start_idx + n_wavelet] = model_wavelet
                
                autocorr_residual = autocorr - model_scaled
                
                try:
                    envelope = np.abs(hilbert(autocorr_residual))
                except:
                    envelope = np.abs(autocorr_residual)
                
                lag_times = np.arange(-max_lag, max_lag + 1) * dt
                if len(lag_times) > 1:
                    envelope_area = np.sum(np.abs(envelope)) * (lag_times[1] - lag_times[0])
                else:
                    envelope_area = 0
                value = envelope_area
            
            return (i, j, value)
            
        except Exception as e:
            print(f"Ошибка для f0={start_freq}, f1={end_freq}: {str(e)}")
            return (i, j, 0)
    
    def calculate_heatmap_in_thread(self):
        """Расчет тепловой карты в отдельном потоке"""
        try:
            start_freq_min = float(self.start_freq_min_var.get())
            start_freq_max = float(self.start_freq_max_var.get())
            start_freq_step = float(self.start_freq_step_var.get())
            end_freq_min = float(self.end_freq_min_var.get())
            end_freq_max = float(self.end_freq_max_var.get())
            end_freq_step = float(self.end_freq_step_var.get())
            heatmap_type = self.heatmap_type_var.get()
            
            if start_freq_min <= 0 or start_freq_max <= 0 or end_freq_min <= 0 or end_freq_max <= 0:
                raise ValueError("Частоты должны быть > 0")
            if start_freq_step <= 0 or end_freq_step <= 0:
                raise ValueError("Шаги частот должны быть > 0")
            if start_freq_min >= start_freq_max:
                raise ValueError("Минимальная начальная частота должна быть меньше максимальной")
            if end_freq_min >= end_freq_max:
                raise ValueError("Минимальная конечная частота должна быть меньше максимальной")
            
            start_freqs = np.arange(start_freq_min, start_freq_max + start_freq_step/2, start_freq_step)
            end_freqs = np.arange(end_freq_min, end_freq_max + end_freq_step/2, end_freq_step)
            
            total_points = len(start_freqs) * len(end_freqs)
            
            if total_points > 25000:
                response = messagebox.askyesno(
                    "Предупреждение",
                    f"Будет рассчитано {total_points} точек.\n"
                    f"Это может занять значительное время.\n"
                    f"Продолжить?"
                )
                if not response:
                    return
            
            matrix = np.zeros((len(end_freqs), len(start_freqs)))
            
            heatmap_type_names = {
                'area': 'Площадь АКФ', 
                'center_freq': 'Центральная частота', 
                'impulse_count': 'Число импульсов',
                'envelope_area': 'Пл. под огиб. АКФ'
            }
            
            self.root.after(0, self.show_progress_window, total_points, heatmap_type_names[heatmap_type])
            
            tasks = []
            for i, start_freq in enumerate(start_freqs):
                for j, end_freq in enumerate(end_freqs):
                    if start_freq < end_freq:
                        tasks.append({
                            'i': i,
                            'j': j,
                            'start_freq': start_freq,
                            'end_freq': end_freq,
                            'fixed_params': self.fixed_params.copy(),
                            'heatmap_type': heatmap_type
                        })
                    else:
                        matrix[j, i] = 0
            
            max_workers = min(8, len(tasks))
            completed = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(self.calculate_single_point, task): task for task in tasks}
                
                for future in concurrent.futures.as_completed(future_to_task):
                    if self.calculation_stopped:
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.root.after(0, self.hide_progress_window)
                        self.root.after(0, messagebox.showinfo, "Остановлено", "Расчет тепловой карты остановлен пользователем")
                        return
                    
                    try:
                        i, j, value = future.result()
                        matrix[j, i] = value
                        completed += 1
                        
                        self.root.after(0, self.update_progress, completed, total_points)
                        
                    except Exception as e:
                        print(f"Ошибка при расчете точки: {e}")
            
            if not self.calculation_stopped:
                self.root.after(0, self.hide_progress_window)
                
                self.current_heatmap_data = (start_freqs, end_freqs, matrix, heatmap_type)
                
                self.root.after(0, self.update_heatmap, start_freqs, end_freqs, matrix, heatmap_type)
                
                heatmap_type_name = heatmap_type_names[heatmap_type]
                self.root.after(0, lambda: self.optimization_window.title(f"Подбор параметров частот - {heatmap_type_name}"))
            
        except Exception as e:
            self.root.after(0, self.hide_progress_window)
            self.root.after(0, messagebox.showerror, "Ошибка расчета", f"Ошибка при построении тепловой карты:\n{str(e)}")
        finally:
            self.calculation_stopped = False
            self.calculation_thread = None
            self.root.after(0, lambda: self.stop_button.config(state='disabled'))
    
    def show_progress_window(self, total_points, heatmap_type_name):
        """Показать окно прогресса"""
        self.progress_window = tk.Toplevel(self.optimization_window)
        self.progress_window.title("Прогресс расчета")
        self.progress_window.geometry("300x120")
        self.progress_window.attributes('-topmost', True)
        
        self.progress_window.update_idletasks()
        x = self.optimization_window.winfo_x() + self.optimization_window.winfo_width() - 320
        y = self.optimization_window.winfo_y() + 50
        self.progress_window.geometry(f"+{x}+{y}")
        
        ttk.Label(self.progress_window, text=f"Расчет {heatmap_type_name}...", 
                 font=('Arial', 10, 'bold')).pack(pady=(20, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, 
                                          maximum=total_points, length=250)
        self.progress_bar.pack(pady=10)
        
        self.stop_progress_button = ttk.Button(self.progress_window, text="Остановить", 
                                             command=self.stop_calculation, width=15)
        self.stop_progress_button.pack(pady=5)
    
    def update_progress(self, completed, total):
        """Обновить прогресс"""
        if hasattr(self, 'progress_var'):
            self.progress_var.set(completed)
            self.progress_window.update()
    
    def hide_progress_window(self):
        """Скрыть окно прогресса"""
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.destroy()
    
    def calculate_heatmap(self):
        """Запуск расчета тепловой карты в отдельном потоке"""
        self.calculation_stopped = False
        self.stop_button.config(state='normal')
        
        self.calculation_thread = threading.Thread(target=self.calculate_heatmap_in_thread, daemon=True)
        self.calculation_thread.start()
    
    def create_impulse_times_only(self, params):
        """Оптимизированное создание только времен импульсов"""
        def temp_frequency_function(t):
            duration = params['duration']
            f0 = params['start_freq']
            f1 = params['end_freq']
            law_type = params['law_type']
            
            if law_type == 'linear':
                return f0 + (f1 - f0) * (t / duration)
            elif law_type == 'quadratic':
                a = (f1 - f0) / (duration**2)
                return a * (t**2) + f0
            elif law_type == 'exponential':
                if f0 <= 0 or f1 <= 0:
                    return f0
                k = math.log(f1 / f0) / duration if f0 > 0 else 0
                return f0 * math.exp(k * t)
            elif law_type == 'hyperbolic':
                if t >= duration:
                    return f1
                term1 = f0**(-2) * (1 - t/duration)
                term2 = f1**(-2) * (t/duration)
                denominator = term1 + term2
                if denominator > 0:
                    return 1.0 / math.sqrt(denominator)
                else:
                    return f0
            else:
                return f0 + (f1 - f0) * (t / duration)
        
        duration = params['duration']
        impulse_times = []
        
        t_current = 0
        iteration_count = 0
        MAX_ITERATIONS = 1000000
        
        while t_current < duration and iteration_count < MAX_ITERATIONS:
            f_current = temp_frequency_function(t_current)
            
            if f_current > 0:
                T = 1.0 / f_current
            else:
                T = duration
                
            MIN_PERIOD = 0.0001
            if T < MIN_PERIOD:
                T = MIN_PERIOD
                
            impulse_times.append(t_current)
            t_current += T
            iteration_count += 1
        
        if iteration_count >= MAX_ITERATIONS:
            print(f"Предупреждение: достигнут предел итераций для f0={params['start_freq']}, f1={params['end_freq']}")
            if len(impulse_times) > 100000:
                impulse_times = impulse_times[:100000]
        
        return impulse_times
    
    def create_impulse_sequence_with_params(self, params):
        """Создание импульсной последовательности с заданными параметрами"""
        def temp_frequency_function(t):
            duration = params['duration']
            f0 = params['start_freq']
            f1 = params['end_freq']
            law_type = params['law_type']
            
            if law_type == 'linear':
                return f0 + (f1 - f0) * (t / duration)
            elif law_type == 'quadratic':
                a = (f1 - f0) / (duration**2)
                return a * (t**2) + f0
            elif law_type == 'exponential':
                if f0 <= 0 or f1 <= 0:
                    return f0
                k = math.log(f1 / f0) / duration if f0 > 0 else 0
                return f0 * math.exp(k * t)
            elif law_type == 'hyperbolic':
                if t >= duration:
                    return f1
                term1 = f0**(-2) * (1 - t/duration)
                term2 = f1**(-2) * (t/duration)
                denominator = term1 + term2
                if denominator > 0:
                    return 1.0 / math.sqrt(denominator)
                else:
                    return f0
            else:
                return f0 + (f1 - f0) * (t / duration)
        
        duration = params['duration']
        dt = params['dt']
        
        MIN_DT = 0.0001
        if dt < MIN_DT:
            dt = MIN_DT
            params['dt'] = dt
        
        impulse_times = []
        impulse_frequencies = []
        
        t_current = 0
        iteration_count = 0
        MAX_ITERATIONS = 1000000
        
        while t_current < duration and iteration_count < MAX_ITERATIONS:
            f_current = temp_frequency_function(t_current)
            
            if f_current > 0:
                T = 1.0 / f_current
            else:
                T = duration
                
            MIN_PERIOD = 0.0001
            if T < MIN_PERIOD:
                T = MIN_PERIOD
                
            impulse_times.append(t_current)
            impulse_frequencies.append(f_current)
            t_current += T
            iteration_count += 1
        
        if iteration_count >= MAX_ITERATIONS:
            print(f"Предупреждение: достигнут предел итераций для f0={params['start_freq']}, f1={params['end_freq']}")
            if len(impulse_times) > 100000:
                impulse_times = impulse_times[:100000]
                impulse_frequencies = impulse_frequencies[:100000]
        
        MAX_IMPULSES = 100000
        if len(impulse_times) > MAX_IMPULSES:
            impulse_times = impulse_times[:MAX_IMPULSES]
            impulse_frequencies = impulse_frequencies[:MAX_IMPULSES]
        
        time = np.arange(0, duration, dt)
        
        MAX_SAMPLES = 500000
        if len(time) > MAX_SAMPLES:
            dt = duration / MAX_SAMPLES
            params['dt'] = dt
            time = np.arange(0, duration, dt)
        
        signal = np.zeros_like(time)
        
        if params['variable_amplitude'] and len(impulse_times) > 0:
            for i, t in enumerate(impulse_times):
                idx = int(t / dt)
                if 0 <= idx < len(signal):
                    amplitude = 1.0 + (i / (len(impulse_times) - 1)) if len(impulse_times) > 1 else 1.0
                    signal[idx] = amplitude
        else:
            for t in impulse_times:
                idx = int(t / dt)
                if 0 <= idx < len(signal):
                    signal[idx] = 1.0
        
        return time, signal, impulse_times, impulse_frequencies
    
    def ricker_wavelet_with_params(self, frequency, params, length=0.1):
        """Создание вейвлета Рикера с заданными параметрами"""
        dt = params['dt']
        
        if dt <= 0:
            dt = 0.001
            params['dt'] = dt
        
        t = np.arange(-length/2, length/2, dt)
        t2 = t ** 2
        wavelet = (1.0 - 2.0 * np.pi**2 * frequency**2 * t2) * np.exp(-np.pi**2 * frequency**2 * t2)
        return wavelet
    
    def compute_envelope_area(self, autocorr, wavelet, dt):
        """Вычисление площади под огибающей АКФ после вычета исходного импульса"""
        model_wavelet = wavelet / np.max(np.abs(wavelet)) if np.max(np.abs(wavelet)) > 0 else wavelet
        
        n_wavelet = len(model_wavelet)
        n_autocorr = len(autocorr)
        
        if n_wavelet > n_autocorr:
            model_scaled = model_wavelet[(n_wavelet - n_autocorr) // 2: (n_wavelet - n_autocorr) // 2 + n_autocorr]
        else:
            model_scaled = np.zeros(n_autocorr)
            start_idx = (n_autocorr - n_wavelet) // 2
            model_scaled[start_idx:start_idx + n_wavelet] = model_wavelet
        
        autocorr_residual = autocorr - model_scaled
        
        try:
            envelope = np.abs(hilbert(autocorr_residual))
        except:
            envelope = np.abs(autocorr_residual)
        
        envelope_area = np.trapz(np.abs(envelope), dx=dt)
        
        return envelope_area, autocorr_residual, envelope
    
    def update_heatmap(self, start_freqs, end_freqs, matrix, heatmap_type='area'):
        """Обновление тепловой карты и палитры"""
        self.ax_heatmap.clear()
        self.ax_colorbar.clear()
        
        heatmap_configs = {
            'area': {
                'title': "Площадь под АКФ",
                'cbar_label': 'Площадь под АКФ',
                'format_str': '.3f'
            },
            'center_freq': {
                'title': "Центральная частота (Гц)",
                'cbar_label': 'Центральная частота (Гц)',
                'format_str': '.1f'
            },
            'impulse_count': {
                'title': "Число импульсов",
                'cbar_label': 'Число импульсов',
                'format_str': '.0f'
            },
            'envelope_area': {
                'title': "Площадь под огибающей АКФ",
                'cbar_label': 'Площадь под огибающей',
                'format_str': '.4f'
            }
        }
        
        config = heatmap_configs.get(heatmap_type, heatmap_configs['area'])
        title = config['title']
        cbar_label = config['cbar_label']
        format_str = config['format_str']
        
        vmin_manual = self.manual_vmin[heatmap_type]
        vmax_manual = self.manual_vmax[heatmap_type]
        
        if vmin_manual is not None:
            vmin = vmin_manual
        else:
            vmin = np.min(matrix[matrix > 0]) if np.any(matrix > 0) else 0
        
        if vmax_manual is not None:
            vmax = vmax_manual
        else:
            vmax = np.max(matrix)
        
        self.update_palette_fields_for_type(heatmap_type)
        
        X, Y = np.meshgrid(start_freqs, end_freqs)
        
        if vmax > vmin:
            norm_values = (matrix - vmin) / (vmax - vmin)
        else:
            norm_values = np.zeros_like(matrix)
        
        cmap = plt.cm.RdYlBu_r
        colors = cmap(norm_values)
        
        self.ax_heatmap.set_facecolor('white')
        
        x_min = start_freqs[0] - (start_freqs[1] - start_freqs[0])/2 if len(start_freqs) > 1 else start_freqs[0] - 0.5
        x_max = start_freqs[-1] + (start_freqs[-1] - start_freqs[-2])/2 if len(start_freqs) > 1 else start_freqs[-1] + 0.5
        y_min = end_freqs[0] - (end_freqs[1] - end_freqs[0])/2 if len(end_freqs) > 1 else end_freqs[0] - 0.5
        y_max = end_freqs[-1] + (end_freqs[-1] - end_freqs[-2])/2 if len(end_freqs) > 1 else end_freqs[-1] + 0.5
        
        self.ax_heatmap.set_xlim(x_min, x_max)
        self.ax_heatmap.set_ylim(y_min, y_max)
        
        self.ax_heatmap.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='gray')
        
        marker_size = 120
        
        for j in range(len(end_freqs)):
            for i in range(len(start_freqs)):
                if matrix[j, i] > 0:
                    color = colors[j, i]
                    self.ax_heatmap.scatter(start_freqs[i], end_freqs[j], 
                                           color=color, s=marker_size, 
                                           edgecolor='none', zorder=3)
                    
                    if len(start_freqs) <= 15 and len(end_freqs) <= 15:
                        value_text = f'{matrix[j, i]:{format_str}}'
                        y_offset = 0.03 * (y_max - y_min)
                        
                        self.ax_heatmap.text(start_freqs[i], end_freqs[j] + y_offset, value_text,
                                           ha='center', va='bottom',
                                           color='black', fontsize=8,
                                           fontweight='bold',
                                           zorder=4)
        
        self.ax_heatmap.set_xlabel('Начальная частота (Гц)', fontsize=10)
        self.ax_heatmap.set_ylabel('Конечная частота (Гц)', fontsize=10)
        self.ax_heatmap.set_title(title, fontsize=11, fontweight='bold', pad=10)
        
        self.ax_heatmap.set_xticks(start_freqs)
        self.ax_heatmap.set_yticks(end_freqs)
        
        if len(start_freqs) > 10:
            self.ax_heatmap.set_xticks(start_freqs[::max(1, len(start_freqs)//10)])
            xtick_labels = [f'{freq:.1f}' for freq in start_freqs[::max(1, len(start_freqs)//10)]]
            self.ax_heatmap.set_xticklabels(xtick_labels, rotation=45, fontsize=8)
        else:
            xtick_labels = [f'{freq:.1f}' for freq in start_freqs]
            self.ax_heatmap.set_xticklabels(xtick_labels, fontsize=8)
        
        if len(end_freqs) > 10:
            self.ax_heatmap.set_yticks(end_freqs[::max(1, len(end_freqs)//10)])
            ytick_labels = [f'{freq:.1f}' for freq in end_freqs[::max(1, len(end_freqs)//10)]]
            self.ax_heatmap.set_yticklabels(ytick_labels, fontsize=8)
        else:
            ytick_labels = [f'{freq:.1f}' for freq in end_freqs]
            self.ax_heatmap.set_yticklabels(ytick_labels, fontsize=8)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        
        cbar = self.fig_colorbar.colorbar(sm, cax=self.ax_colorbar)
        cbar.set_label(cbar_label, fontsize=10)
        
        self.ax_heatmap.tick_params(axis='both', labelsize=8)
        self.ax_colorbar.tick_params(labelsize=8)
        
        self.fig_heatmap.tight_layout()
        self.fig_colorbar.tight_layout()
        self.canvas_heatmap.draw()
        self.canvas_colorbar.draw()
    
    def on_law_type_change(self):
        """Обработка изменения типа закона"""
        self.update_plots()
    
    def on_var_amp_change(self):
        """Обработка изменения опции переменной амплитуды"""
        self.update_plots()
    
    def get_parameters(self):
        """Получение параметров из полей ввода с проверкой"""
        try:
            params = {
                'ricker_freq': float(self.ricker_freq_entry.get()),
                'duration': float(self.duration_entry.get()),
                'start_freq': float(self.start_freq_entry.get()),
                'end_freq': float(self.end_freq_entry.get()),
                'dt': 0.001,
                'max_lag': 500,
                'law_type': self.law_type_var.get(),
                'variable_amplitude': bool(self.var_amp_var.get())
            }
            
            if params['ricker_freq'] <= 0 or params['duration'] <= 0:
                raise ValueError("Частота и длительность должны быть > 0")
            if params['start_freq'] <= 0 or params['end_freq'] <= 0:
                raise ValueError("Частоты должны быть > 0")
            if params['start_freq'] >= params['end_freq']:
                raise ValueError("Начальная частота должна быть меньше конечной")
            
            self.params = params
            return True
            
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректные параметры:\n{str(e)}")
            return False
    
    def ricker_wavelet(self, frequency, length=0.1):
        """Создание вейвлета Рикера"""
        dt = self.params['dt']
        t = np.arange(-length/2, length/2, dt)
        t2 = t ** 2
        wavelet = (1.0 - 2.0 * np.pi**2 * frequency**2 * t2) * np.exp(-np.pi**2 * frequency**2 * t2)
        return wavelet
    
    def scale_compensation_coefficients(self, duration, f0, f1):
        """Масштабирование компенсационных коэффициентов для заданных параметров"""
        base_durations = [10, 20, 40, 80]
        
        if duration <= base_durations[0]:
            idx1, idx2 = 0, 0
            weight = 0.0
        elif duration >= base_durations[-1]:
            idx1, idx2 = -1, -1
            weight = 1.0
        else:
            for i in range(len(base_durations)-1):
                if base_durations[i] <= duration <= base_durations[i+1]:
                    idx1, idx2 = i, i+1
                    weight = (duration - base_durations[i]) / (base_durations[i+1] - base_durations[i])
                    break
        
        coeffs1 = self.compensation_coefficients[idx1]
        coeffs2 = self.compensation_coefficients[idx2]
        
        scaled_coeffs = []
        for c1, c2 in zip(coeffs1, coeffs2):
            scaled_coeffs.append(c1 * (1-weight) + c2 * weight)
        
        e_offset = f0 - scaled_coeffs[4]
        scaled_coeffs[4] = f0
        
        current_end_value = (scaled_coeffs[0] * (duration**4) + 
                            scaled_coeffs[1] * (duration**3) + 
                            scaled_coeffs[2] * (duration**2) + 
                            scaled_coeffs[3] * duration + 
                            scaled_coeffs[4])
        
        diff = f1 - current_end_value
        
        total_poly = 0
        contributions = []
        
        contributions.append(scaled_coeffs[0] * (duration**4))
        contributions.append(scaled_coeffs[1] * (duration**3))
        contributions.append(scaled_coeffs[2] * (duration**2))
        contributions.append(scaled_coeffs[3] * duration)
        
        total_poly = sum(contributions)
        
        if abs(total_poly) > 1e-10:
            for i in range(4):
                if abs(contributions[i]) > 1e-10:
                    scale_factor = 1 + (diff * contributions[i] / total_poly) / contributions[i]
                    scaled_coeffs[i] *= scale_factor
        
        return scaled_coeffs
    
    def frequency_function(self, t):
        """Вычисление частоты в зависимости от выбранного закона"""
        duration = self.params['duration']
        f0 = self.params['start_freq']
        f1 = self.params['end_freq']
        law_type = self.params['law_type']
        
        if law_type == 'linear':
            return f0 + (f1 - f0) * (t / duration)
        
        elif law_type == 'quadratic':
            a = (f1 - f0) / (duration**2)
            return a * (t**2) + f0
        
        elif law_type == 'exponential':
            if f0 <= 0 or f1 <= 0:
                return f0
            k = math.log(f1 / f0) / duration if f0 > 0 else 0
            return f0 * math.exp(k * t)
        
        elif law_type == 'compensation':
            coeffs = self.scale_compensation_coefficients(duration, f0, f1)
            
            return (coeffs[0] * (t**4) + 
                    coeffs[1] * (t**3) + 
                    coeffs[2] * (t**2) + 
                    coeffs[3] * t + 
                    coeffs[4])
        
        elif law_type == 'hyperbolic':
            if t >= duration:
                return f1
            
            term1 = f0**(-2) * (1 - t/duration)
            term2 = f1**(-2) * (t/duration)
            denominator = term1 + term2
            
            if denominator > 0:
                return 1.0 / math.sqrt(denominator)
            else:
                return f0
        
        else:
            return f0
    
    def create_hyperbolic_sequence_analytical(self):
        """Создание гиперболической последовательности аналитически"""
        duration = self.params['duration']
        f0 = self.params['start_freq']
        f1 = self.params['end_freq']
        
        if duration <= 0 or f0 <= 0 or f1 <= 0:
            return [], []
        
        T0 = 1.0 / f0
        T1 = 1.0 / f1
        
        N_approx = int(duration / ((T0 + T1) / 2))
        
        dT = (T0 - T1) / (N_approx - 1) if N_approx > 1 else 0
        
        impulse_times = []
        impulse_frequencies = []
        
        n = 0
        iteration_count = 0
        MAX_ITERATIONS = 1000000
        
        while iteration_count < MAX_ITERATIONS:
            t_n = n * T0 - n * (n - 1) / 2 * dT
            
            if t_n > duration:
                break
                
            if t_n >= 0:
                if n == 0:
                    f_n = f0
                elif n == N_approx - 1:
                    f_n = f1
                else:
                    f_n = 1.0 / (T0 - n * dT)
                
                impulse_times.append(t_n)
                impulse_frequencies.append(f_n)
            
            n += 1
            iteration_count += 1
        
        if iteration_count >= MAX_ITERATIONS:
            print(f"Предупреждение: достигнут предел итераций для гиперболической последовательности")
            if len(impulse_times) > 100000:
                impulse_times = impulse_times[:100000]
                impulse_frequencies = impulse_frequencies[:100000]
        
        return impulse_times, impulse_frequencies
    
    def create_impulse_sequence(self):
        """Создание импульсной последовательности"""
        duration = self.params['duration']
        dt = self.params['dt']
        
        if self.params['law_type'] == 'hyperbolic':
            impulse_times, impulse_frequencies = self.create_hyperbolic_sequence_analytical()
        else:
            impulse_times = []
            impulse_frequencies = []
            
            t_current = 0
            iteration_count = 0
            MAX_ITERATIONS = 1000000
            
            while t_current < duration and iteration_count < MAX_ITERATIONS:
                f_current = self.frequency_function(t_current)
                
                if f_current > 0:
                    T = 1.0 / f_current
                else:
                    T = duration
                    
                impulse_times.append(t_current)
                impulse_frequencies.append(f_current)
                t_current += T
                iteration_count += 1
            
            if iteration_count >= MAX_ITERATIONS:
                print(f"Предупреждение: достигнут предел итераций при создании последовательности")
                if len(impulse_times) > 100000:
                    impulse_times = impulse_times[:100000]
                    impulse_frequencies = impulse_frequencies[:100000]
        
        time = np.arange(0, duration, dt)
        
        MAX_SAMPLES = 500000
        if len(time) > MAX_SAMPLES:
            dt = duration / MAX_SAMPLES
            self.params['dt'] = dt
            time = np.arange(0, duration, dt)
        
        signal = np.zeros_like(time)
        
        if self.params['variable_amplitude'] and len(impulse_times) > 0:
            for i, t in enumerate(impulse_times):
                idx = int(t / dt)
                if idx < len(signal):
                    amplitude = 1.0 + (i / (len(impulse_times) - 1)) if len(impulse_times) > 1 else 1.0
                    signal[idx] = amplitude
        else:
            for t in impulse_times:
                idx = int(t / dt)
                if idx < len(signal):
                    signal[idx] = 1.0
        
        return time, signal, impulse_times, impulse_frequencies
    
    def compute_autocorrelation(self, signal):
        """Вычисление автокорреляционной функции"""
        max_lag = self.params['max_lag']
        n = len(signal)
        
        if n > 200000:
            max_lag = min(max_lag, 200)
        
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n-1-max_lag:n+max_lag]
        lags = np.arange(-max_lag, max_lag + 1)
        
        max_val = np.max(np.abs(autocorr))
        if max_val > 0:
            autocorr = autocorr / max_val
        
        return lags, autocorr
    
    def compute_spectrum(self, autocorr):
        """Вычисление спектра АКФ"""
        dt = self.params['dt']
        n = len(autocorr)
        
        if n > 100000:
            autocorr = autocorr[:100000]
            n = len(autocorr)
        
        spectrum = np.fft.fft(autocorr)
        freq = np.fft.fftfreq(n, dt)
        
        pos_freq = freq[:n//2]
        pos_spectrum = np.abs(spectrum[:n//2])
        
        if np.max(pos_spectrum) > 0:
            pos_spectrum = pos_spectrum / np.max(pos_spectrum)
        
        return pos_freq, pos_spectrum
    
    def update_frequency_plot(self):
        """Обновление графика роста частоты"""
        self.ax_freq.clear()
        
        duration = self.params['duration']
        dt = duration / 200
        time_points = np.arange(0, duration + dt, dt)
        freq_points = [self.frequency_function(t) for t in time_points]
        
        law_colors = {
            'linear': '#FF6B6B',
            'quadratic': '#4ECDC4',
            'exponential': '#45B7D1',
            'compensation': '#96CEB4',
            'hyperbolic': '#FFA726'
        }
        
        color = law_colors.get(self.params['law_type'], '#45B7D1')
        
        self.ax_freq.plot(time_points, freq_points, color=color, linewidth=2)
        
        self.ax_freq.set_xlabel('Время (сек)', fontsize=9)
        self.ax_freq.set_ylabel('Частота (Гц)', fontsize=9)
        self.ax_freq.grid(True, alpha=0.3)
        
        self.ax_freq.set_xlim(0, duration)
        f_min = min(self.params['start_freq'], self.params['end_freq'])
        f_max = max(self.params['start_freq'], self.params['end_freq'])
        
        f_range = f_max - f_min
        padding = 0.1 * f_range if f_range > 0 else 1.0
        self.ax_freq.set_ylim(max(0.1, f_min - padding), f_max + padding)
        
        formulas = {
            'linear': 'f(t) = f₀ + (f₁ - f₀)·t/T',
            'quadratic': 'f(t) = a·t² + f₀',
            'exponential': 'f(t) = f₀·exp(k·t)',
            'compensation': 'f(t) = полином 4-й ст.',
            'hyperbolic': 'f(t) = 1/√[f₀⁻²(1-t/T)+f₁⁻²(t/T)]'
        }
        formula = formulas.get(self.params['law_type'], '')
        
        self.ax_freq.text(0.02, 0.98, formula, 
                         transform=self.ax_freq.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         fontsize=8)
        
        self.fig_freq.tight_layout()
        self.canvas_freq.draw()
    
    def update_plots(self):
        """Обновление всех графиков"""
        if not self.get_parameters():
            return
        
        try:
            law_type = self.params['law_type']
            
            # ЛЕВЫЙ ВЕРХНИЙ: Импульсная последовательность
            time, signal, impulse_times, impulse_freqs = self.create_impulse_sequence()
            
            # ПРАВЫЙ ВЕРХНИЙ: Свертка с вейвлетом Рикера
            wavelet = self.ricker_wavelet(self.params['ricker_freq'])
            convolution = np.convolve(signal, wavelet, mode='same')
            
            # ЛЕВЫЙ НИЖНИЙ: Автокорреляционная функция
            lags, autocorr = self.compute_autocorrelation(convolution)
            lag_times = lags * self.params['dt']
            
            area = np.sum(np.abs(autocorr)) * (lag_times[1] - lag_times[0])
            
            envelope_area, autocorr_residual, envelope = self.compute_envelope_area(
                autocorr, wavelet, self.params['dt']
            )
            
            # ПРАВЫЙ НИЖНИЙ: Спектр
            spectrum_freq, spectrum = self.compute_spectrum(autocorr)
            
            self.ax_impulse.clear()
            self.ax_convolution.clear()
            self.ax_autocorr.clear()
            self.ax_spectrum.clear()
            
            # ЛЕВЫЙ ВЕРХНИЙ ГРАФИК: Импульсная последовательность
            show_duration = min(5.0, self.params['duration'])
            show_mask = time <= show_duration
            show_time = time[show_mask]
            show_signal = signal[show_mask]
            
            impulse_indices = np.where(show_signal > 0)[0]
            
            if len(impulse_indices) > 0:
                impulse_positions = show_time[impulse_indices]
                impulse_amplitudes = show_signal[impulse_indices]
                
                if self.params['variable_amplitude']:
                    colors = plt.cm.RdYlBu((impulse_amplitudes - 1.0))
                    
                    for pos, amp, color in zip(impulse_positions, impulse_amplitudes, colors):
                        self.ax_impulse.plot([pos, pos], [0, amp], 
                                           color=color, linewidth=1.5, alpha=0.7)
                        self.ax_impulse.scatter([pos], [amp], color=color, s=20, alpha=0.8)
                else:
                    for pos in impulse_positions:
                        self.ax_impulse.axvline(x=pos, color='#006400', alpha=0.7,
                                              linewidth=1.2, ymin=0.45, ymax=0.55)
                
                self.ax_impulse.step(show_time, show_signal, where='post', 
                                   color='green', linewidth=0.5, alpha=0.3)
            
            self.ax_impulse.set_xlim(0, show_duration)
            
            if self.params['variable_amplitude']:
                self.ax_impulse.set_ylim(-0.2, 2.5)
                ylabel = 'Амплитуда'
            else:
                self.ax_impulse.set_ylim(-0.2, 1.5)
                ylabel = 'Сигнал (0/1)'
            
            law_names = {
                'linear': 'Линейный',
                'quadratic': 'Квадратичный',
                'exponential': 'Экспоненциальный',
                'compensation': 'Компенсационный',
                'hyperbolic': 'Гиперболический'
            }
            law_name = law_names.get(self.params['law_type'], self.params['law_type'])
            
            amp_status = " (пер.ампл.)" if self.params['variable_amplitude'] else ""
            self.ax_impulse.set_title(f'Импульсная последовательность{amp_status} ({law_name})', 
                                     fontsize=11, fontweight='bold')
            self.ax_impulse.set_xlabel('Время (сек)')
            self.ax_impulse.set_ylabel(ylabel)
            self.ax_impulse.grid(True, alpha=0.3, linestyle='--')
            self.ax_impulse.axhline(y=0, color='black', alpha=0.5, linewidth=0.5)
            
            if not self.params['variable_amplitude']:
                self.ax_impulse.axhline(y=1, color='red', alpha=0.3, linestyle=':', linewidth=1)
            
            total_impulses = len(impulse_times)
            
            info_text = f'Всего импульсов: {total_impulses}\n'
            info_text += f'Нач. частота: {self.params["start_freq"]:.1f} Гц\n'
            info_text += f'Кон. частота: {self.params["end_freq"]:.1f} Гц'
            
            if self.params['variable_amplitude'] and total_impulses > 0:
                min_amp = np.min(signal[signal > 0])
                max_amp = np.max(signal[signal > 0])
                info_text += f'\nАмплитуда: {min_amp:.1f}→{max_amp:.1f}'
            
            self.ax_impulse.text(0.02, 0.98, info_text, 
                               transform=self.ax_impulse.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                               fontsize=8)
            
            # ПРАВЫЙ ВЕРХНИЙ ГРАФИК: Свертка
            convolution_duration = min(1.0, self.params['duration'])
            conv_mask = time <= convolution_duration
            conv_time = time[conv_mask]
            conv_values = convolution[conv_mask]
            
            conv_impulse_mask = (np.array(impulse_times) <= convolution_duration)
            conv_impulse_times = np.array(impulse_times)[conv_impulse_mask]
            
            self.ax_convolution.plot(conv_time, conv_values, color='purple', linewidth=1.5, alpha=0.8)
            self.ax_convolution.fill_between(conv_time, 0, conv_values, color='purple', alpha=0.2)
            
            if len(conv_impulse_times) > 0:
                for imp_time in conv_impulse_times:
                    self.ax_convolution.axvline(x=imp_time, color='red', alpha=0.4, 
                                               linestyle='--', linewidth=0.8)
            
            self.ax_convolution.set_xlim(0, convolution_duration)
            
            if len(conv_values) > 0:
                y_min, y_max = np.min(conv_values), np.max(conv_values)
                y_range = y_max - y_min
                if y_range > 0:
                    self.ax_convolution.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
            
            amp_status = " (пер.ампл.)" if self.params['variable_amplitude'] else ""
            self.ax_convolution.set_title(f'Свертка с вейвлетом Рикера{amp_status} ({self.params["ricker_freq"]} Гц)', 
                                        fontsize=11, fontweight='bold')
            self.ax_convolution.set_xlabel('Время (сек)')
            self.ax_convolution.set_ylabel('Амплитуда')
            self.ax_convolution.grid(True, alpha=0.3, linestyle='--')
            self.ax_convolution.axhline(y=0, color='black', alpha=0.5, linewidth=0.5)
            
            conv_info = f'Частота Рикера: {self.params["ricker_freq"]} Гц\n'
            conv_info += f'Макс. амплитуда: {np.max(np.abs(conv_values)):.3f}\n'
            conv_info += f'Импульсов в области: {len(conv_impulse_times)}'
            
            self.ax_convolution.text(0.02, 0.98, conv_info, 
                                   transform=self.ax_convolution.transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                                   fontsize=8)
            
            # ЛЕВЫЙ НИЖНИЙ ГРАФИК: АКФ с огибающей
            self.ax_autocorr.plot(lag_times, autocorr, color='#4ECDC4', linewidth=2, label='АКФ')
            self.ax_autocorr.fill_between(lag_times, 0, autocorr, color='#4ECDC4', alpha=0.3)
            
            self.ax_autocorr.plot(lag_times, envelope, color='red', linewidth=1.5, 
                                 linestyle='--', alpha=0.8, label='Огибающая АКФ')
            
            self.ax_autocorr.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            
            amp_status = " (пер.ампл.)" if self.params['variable_amplitude'] else ""
            self.ax_autocorr.set_title(f'Автокорреляция свертки{amp_status}', fontsize=12)
            self.ax_autocorr.set_xlabel('Лаг (сек)')
            self.ax_autocorr.set_ylabel('Нормализованная автокорреляция')
            self.ax_autocorr.set_xlim(-0.5, 0.5)
            self.ax_autocorr.grid(True, alpha=0.3)
            self.ax_autocorr.legend(loc='upper right', fontsize=8)
            
            acf_info = f'Площадь АКФ: {area:.3f}\n'
            acf_info += f'Пл. под огиб. АКФ: {envelope_area:.4f}'
            
            self.ax_autocorr.text(0.02, 0.98, acf_info, 
                                transform=self.ax_autocorr.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                fontsize=8)
            
            # ПРАВЫЙ НИЖНИЙ ГРАФИК: Спектр
            self.ax_spectrum.plot(spectrum_freq, spectrum, color='#6c5ce7', linewidth=2)
            self.ax_spectrum.fill_between(spectrum_freq, 0, spectrum, color='#6c5ce7', alpha=0.3)
            
            max_freq = min(500, 1/(2*self.params['dt']))
            freq_mask = spectrum_freq <= max_freq
            
            if len(spectrum_freq[freq_mask]) > 1:
                self.ax_spectrum.set_xlim(0, max_freq)
                
                spectrum_freq_masked = spectrum_freq[freq_mask]
                spectrum_masked = spectrum[freq_mask]
                dominant_freq_idx = np.argmax(spectrum_masked)
                dominant_freq = spectrum_freq_masked[dominant_freq_idx]
                dominant_amp = spectrum_masked[dominant_freq_idx]
                
                self.ax_spectrum.axvline(x=dominant_freq, color='red', linestyle='--', alpha=0.7, linewidth=1)
                
                if dominant_amp > 0.1:
                    spec_info = f'Доминирующая частота:\n{dominant_freq:.1f} Гц ({dominant_amp:.2f})'
                    
                    self.ax_spectrum.text(0.02, 0.98, spec_info, 
                                        transform=self.ax_spectrum.transAxes, verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                        fontsize=8)
            
            amp_status = " (пер.ампл.)" if self.params['variable_amplitude'] else ""
            self.ax_spectrum.set_title(f'Спектр автокорреляционной функции{amp_status}', fontsize=12)
            self.ax_spectrum.set_xlabel('Частота (Гц)')
            self.ax_spectrum.set_ylabel('Амплитуда (норм.)')
            self.ax_spectrum.grid(True, alpha=0.3)
            
            # Обновление графика роста частоты
            self.update_frequency_plot()
            
            self.current_data = {
                'time': time,
                'signal': signal,
                'impulse_times': impulse_times,
                'impulse_frequencies': impulse_freqs,
                'convolution': convolution,
                'lag_times': lag_times,
                'autocorr': autocorr,
                'autocorr_residual': autocorr_residual,
                'envelope': envelope,
                'envelope_area': envelope_area,
                'spectrum_freq': spectrum_freq,
                'spectrum': spectrum,
                'area': area,
                'law_type': self.params['law_type'],
                'law_name': law_name,
                'variable_amplitude': self.params['variable_amplitude']
            }
            
            self.fig_freq.tight_layout()
            self.fig_top.tight_layout()
            self.fig_bottom.tight_layout()
            
            self.canvas_freq.draw()
            self.canvas_top.draw()
            self.canvas_bottom.draw()
            
        except Exception as e:
            messagebox.showerror("Ошибка расчета", f"Ошибка при построении графиков:\n{str(e)}")
    
    def save_autocorrelation(self):
        """Сохранение АКФ в текстовый файл"""
        if not hasattr(self, 'current_data'):
            messagebox.showwarning("Нет данных", "Сначала постройте графики")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
            initialfile=f"АКФ_{self.params['duration']}сек_{self.params['law_type']}{'_varamp' if self.params['variable_amplitude'] else ''}.txt"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Автокорреляционная функция\n")
                f.write("=" * 70 + "\n")
                f.write(f"Параметры:\n")
                f.write(f"  Закон изменения частоты: {self.current_data['law_name']}\n")
                if self.params['variable_amplitude']:
                    f.write(f"  Амплитуда импульсов: линейно от 1 до 2\n")
                f.write(f"  Частота вейвлета Рикера: {self.params['ricker_freq']} Гц\n")
                f.write(f"  Длительность последовательности: {self.params['duration']} сек\n")
                f.write(f"  Начальная частота импульсов: {self.params['start_freq']} Гц\n")
                f.write(f"  Конечная частота импульсов: {self.params['end_freq']} Гц\n")
                f.write(f"  Площадь под графиком АКФ: {self.current_data['area']:.4f}\n")
                f.write(f"  Площадь под огибающей АКФ: {self.current_data['envelope_area']:.6f}\n")
                f.write("=" * 70 + "\n")
                f.write("Лаг(мс)\tЛаг(сек)\tАКФ(норм.)\tОстаток АКФ\tОгибающая\n")
                f.write("-" * 60 + "\n")
                
                for lag_time, acf, residual, env in zip(
                    self.current_data['lag_times'], 
                    self.current_data['autocorr'], 
                    self.current_data['autocorr_residual'],
                    self.current_data['envelope']
                ):
                    lag_ms = lag_time * 1000
                    f.write(f"{lag_ms:.3f}\t{lag_time:.6f}\t{acf:.6f}\t{residual:.6f}\t{env:.6f}\n")
            
            messagebox.showinfo("Сохранение", f"АКФ успешно сохранена в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def save_impulse_times(self):
        """Сохранение времен ударов в текстовый файл"""
        if not hasattr(self, 'current_data'):
            messagebox.showwarning("Нет данных", "Сначала постройте графики")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
            initialfile=f"Времена_ударов_{self.params['duration']}сек_{self.params['law_type']}{'_varamp' if self.params['variable_amplitude'] else ''}.txt"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Времена ударов (импульсная последовательность)\n")
                f.write("=" * 70 + "\n")
                f.write(f"Параметры:\n")
                f.write(f"  Закон изменения частоты: {self.current_data['law_name']}\n")
                if self.params['variable_amplitude']:
                    f.write(f"  Амплитуда импульсов: линейно от 1 до 2\n")
                f.write(f"  Длительность последовательности: {self.params['duration']} сек\n")
                f.write(f"  Начальная частота: {self.params['start_freq']} Гц\n")
                f.write(f"  Конечная частота: {self.params['end_freq']} Гц\n")
                f.write(f"  Общее число импульсов: {len(self.current_data['impulse_times'])}\n")
                f.write(f"  Частота вейвлета Рикера: {self.params['ricker_freq']} Гц\n")
                f.write("=" * 70 + "\n")
                f.write("№\tВремя(сек)\tВремя(мс)\tЧастота(Гц)\tПериод(мс)\tАмплитуда\tДискретный_сигнал\n")
                f.write("-" * 70 + "\n")
                
                dt = self.params['dt']
                
                for i, (imp_time, imp_freq) in enumerate(zip(self.current_data['impulse_times'], 
                                                           self.current_data['impulse_frequencies'])):
                    period = 1.0 / imp_freq if imp_freq > 0 else 0
                    
                    sample_idx = int(imp_time / dt)
                    if sample_idx < len(self.current_data['signal']):
                        disc_signal = self.current_data['signal'][sample_idx]
                        amplitude = disc_signal if disc_signal > 0 else 0
                    else:
                        disc_signal = 0
                        amplitude = 0
                    
                    f.write(f"{i+1}\t{imp_time:.6f}\t{imp_time*1000:.3f}\t{imp_freq:.3f}\t{period*1000:.3f}\t{amplitude:.2f}\t{disc_signal}\n")
            
            messagebox.showinfo("Сохранение", f"Времена ударов успешно сохранены в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def save_convolution(self):
        """Сохранение результата свертки в текстовый файл"""
        if not hasattr(self, 'current_data'):
            messagebox.showwarning("Нет данных", "Сначала постройте графики")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
            initialfile=f"Свертка_{self.params['duration']}сек_{self.params['law_type']}{'_varamp' if self.params['variable_amplitude'] else ''}.txt"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Результат свертки импульсной последовательности и вейвлета Рикера\n")
                f.write("=" * 70 + "\n")
                f.write(f"Параметры:\n")
                f.write(f"  Закон изменения частоты: {self.current_data['law_name']}\n")
                if self.params['variable_amplitude']:
                    f.write(f"  Амплитуда импульсов: линейно от 1 до 2\n")
                f.write(f"  Частота вейвлета Рикера: {self.params['ricker_freq']} Гц\n")
                f.write(f"  Длительность последовательности: {self.params['duration']} сек\n")
                f.write(f"  Начальная частота импульсов: {self.params['start_freq']} Гц\n")
                f.write(f"  Конечная частота импульсов: {self.params['end_freq']} Гц\n")
                f.write(f"  Шаг по времени: {self.params['dt']} сек\n")
                f.write(f"  Число отсчетов: {len(self.current_data['time'])}\n")
                f.write(f"  Максимальная амплитуда свертки: {np.max(np.abs(self.current_data['convolution'])):.6f}\n")
                f.write("=" * 70 + "\n")
                f.write("Время(сек)\tВремя(мс)\tАмплитуда\n")
                f.write("-" * 35 + "\n")
                
                for time_val, conv_val in zip(self.current_data['time'], self.current_data['convolution']):
                    f.write(f"{time_val:.6f}\t{time_val*1000:.3f}\t{conv_val:.6f}\n")
            
            messagebox.showinfo("Сохранение", f"Свертка успешно сохранена в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def save_all_parameters(self):
        """Сохранение всех параметров в JSON файл"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON файлы", "*.json"), ("Все файлы", "*.*")],
            initialfile="parameters.json"
        )
        
        if not file_path:
            return
        
        try:
            save_data = {
                'parameters': self.params,
                'calculated_data': {
                    'num_impulses': len(self.current_data['impulse_times']),
                    'area': float(self.current_data['area']),
                    'envelope_area': float(self.current_data['envelope_area']),
                    'max_convolution': float(np.max(np.abs(self.current_data['convolution']))),
                    'duration_seconds': float(self.params['duration']),
                    'law_type': self.current_data['law_type'],
                    'law_name': self.current_data['law_name'],
                    'variable_amplitude': self.current_data['variable_amplitude']
                } if hasattr(self, 'current_data') else {}
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            
            messagebox.showinfo("Сохранение", f"Параметры успешно сохранены в файл:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")
    
    def reset_to_defaults(self):
        """Сброс параметров к значениям по умолчанию"""
        self.ricker_freq_var.set(self.default_params['ricker_freq'])
        self.duration_var.set(self.default_params['duration'])
        self.start_freq_var.set(self.default_params['start_freq'])
        self.end_freq_var.set(self.default_params['end_freq'])
        self.law_type_var.set(self.default_params['law_type'])
        self.var_amp_var.set(self.default_params['variable_amplitude'])
        
        self.update_plots()

def main():
    root = tk.Tk()
    app = AutocorrelationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()