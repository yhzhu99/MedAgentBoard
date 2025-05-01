# TODO: 检查file path的更新

import json
import random
import os
import argparse

def process_json_and_sample(input_filepath, output_filepath, sample_size=100):
    """
    读取JSON文件，为每条记录添加ID，然后随机抽样指定数量的记录到新文件。

    Args:
        input_filepath (str): 输入JSON文件的路径。
        output_filepath (str): 输出JSON文件的路径。
        sample_size (int): 需要抽样的记录数量。
    """
    random.seed(42)  # 设置随机种子以确保可重复性
    # --- 1. 读取输入 JSON 文件 ---
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            original_data = json.load(f_in)
        print(f"成功读取文件: {input_filepath}")
    except FileNotFoundError:
        print(f"错误：找不到输入文件 '{input_filepath}'")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{input_filepath}' 不是有效的 JSON 格式。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return

    # 检查数据是否是列表格式
    if not isinstance(original_data, list):
        print(f"错误：文件 '{input_filepath}' 的顶层结构不是 JSON 列表。")
        return

    # --- 2. 为每条记录添加 ID ---
    data_with_ids = []
    for index, item in enumerate(original_data):
        # 确保 item 是字典类型，以防 JSON 列表里有非对象元素
        if isinstance(item, dict):
            # 创建一个新字典或直接修改 item，这里选择直接修改
            item['id'] = index + 1
            data_with_ids.append(item)
        else:
            print(f"警告：跳过索引 {index} 处的非对象元素: {item}")

    total_items = len(data_with_ids)
    print(f"已为 {total_items} 条有效记录添加 ID。")

    # --- 3. 随机抽样 ---
    if total_items == 0:
        print("警告：没有有效的记录可供抽样。输出文件将为空列表。")
        sampled_data = []
    elif total_items < sample_size:
        print(f"警告：记录总数 ({total_items}) 少于要求的抽样数量 ({sample_size})。将抽取所有 {total_items} 条记录。")
        # 直接使用所有数据，或者可以用 random.sample(data_with_ids, total_items) 效果一样
        sampled_data = data_with_ids
    else:
        print(f"正在从 {total_items} 条记录中随机抽取 {sample_size} 条...")
        sampled_data = random.sample(data_with_ids, sample_size)
        print(f"成功抽取 {len(sampled_data)} 条记录。")

    # --- 4. 将抽样结果写入新 JSON 文件 ---
    try:
        # 确保输出目录存在（如果输出路径包含目录）
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            # indent=4 使输出的 JSON 文件格式更美观易读
            # ensure_ascii=False 确保中文字符能正确写入，而不是被转义成 \uXXXX
            json.dump(sampled_data, f_out, indent=4, ensure_ascii=False)
        print(f"抽样结果已成功写入到: {output_filepath}")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset')
    args = parser.parse_args()

    for dataset in os.listdir(args.data_path):
        dataset_path = os.path.join(args.data_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for split in os.listdir(dataset_path):
            if split == "test.json":
                input_json_file = os.path.join(args.data_path, dataset, split)
                output_json_file = os.path.join('processed', dataset, split)
                number_to_sample = 100

                process_json_and_sample(input_json_file, output_json_file, number_to_sample)