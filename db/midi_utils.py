#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from music21 import converter


def fix_midi_file(input_path, output_path):
    """
    使用 music21 读取并修复 MIDI 文件，然后保存到 output_path。

    参数:
      input_path: 原始 MIDI 文件路径
      output_path: 输出的修复后 MIDI 文件路径

    返回:
      修复成功则返回 True，否则返回 False
    """
    try:
        # 解析 MIDI 文件
        score = converter.parse(input_path)
        # 将解析后的 score 重新写入为标准 MIDI 文件
        score.write('midi', fp=output_path)
        print(f"修复成功：{input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"修复 {input_path} 失败：{e}")
        return False


def batch_fix_midi(input_dir, output_dir):
    """
    批量修复 MIDI 文件：
      递归遍历 input_dir（包括所有子目录），对扩展名为 .mid 或 .midi 的文件使用 music21 修复，
      并将修复后的文件保存到 output_dir，保持原有目录结构。
    """
    total_files = 0
    fixed_files = 0
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.mid', '.midi')):
                total_files += 1
                input_file = os.path.join(root, filename)
                # 保持相对目录结构
                rel_dir = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, filename)
                if fix_midi_file(input_file, output_file):
                    fixed_files += 1
    print(f"总共处理 {total_files} 个文件，其中成功修复 {fixed_files} 个。")


if __name__ == '__main__':
    # 假设本脚本位于 db 目录中，那么 MIDI 数据存放在 db/clean_midi 中
    base_dir = os.path.dirname(__file__)
    # 输入目录：db/clean_midi（请根据实际情况调整）
    input_dir = os.path.join(base_dir, 'clean_midi')
    # 输出目录：例如固定为 db/fixed_midi
    output_dir = os.path.join(base_dir, 'fixed_midi')
    print("开始批量修复 MIDI 文件：")
    print("输入目录：", input_dir)
    print("输出目录：", output_dir)
    batch_fix_midi(input_dir, output_dir)
