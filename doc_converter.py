"""
文档格式转换工具 - 将.doc文件批量转换为.docx格式
请求管理员权限，转换后文件替换到原位置，备份原始文件
"""
import os
import sys
import ctypes
import shutil
import traceback
from pathlib import Path
import time
import re
import tempfile
import argparse


def is_admin():
    """检查当前是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False


def request_admin_and_restart():
    """
    请求管理员权限并重启程序
    """
    # 获取当前脚本路径
    script_path = os.path.abspath(sys.argv[0])
    work_dir = os.path.dirname(script_path)

    # 构建参数
    if script_path.endswith('.py'):
        executable = sys.executable
        params = script_path + ' ' + ' '.join(sys.argv[1:])  # 保留原有命令行参数
    else:
        executable = script_path
        params = " ".join(sys.argv[1:])

    # 使用ShellExecuteW请求管理员权限
    result = ctypes.windll.shell32.ShellExecuteW(
        None,            # 父窗口句柄
        "runas",         # 操作：以管理员身份运行
        executable,      # 程序路径
        params,          # 参数
        work_dir,        # 工作目录
        1                # 显示方式：1=正常窗口
    )

    sys.exit(0)  # 退出当前非管理员进程


def check_win32com():
    """检查并安装必要的库"""
    try:
        import pythoncom
        from win32com import client as wc
        return True, pythoncom, wc
    except ImportError:
        print(" 未安装必要的库: pywin32")
        print("正在尝试安装...")

        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])

            import pythoncom
            from win32com import client as wc
            print(" pywin32安装成功")
            return True, pythoncom, wc
        except Exception as e:
            print(f" 安装失败: {e}")
            print("请手动运行: pip install pywin32")
            return False, None, None


def sanitize_filename(filename):
    """
    清理文件名，移除可能导致问题的字符
    """
    # 移除文件扩展名
    name, ext = os.path.splitext(filename)

    # 替换可能导致问题的字符
    # Windows不允许的字符: < > : " / \ | ? *
    invalid_chars = r'[<>:"/\\|?*]'
    name = re.sub(invalid_chars, '_', name)

    # 清理开头和结尾的空格和点
    name = name.strip('. ')

    # 限制文件名长度（Windows限制255字符）
    if len(name) > 100:
        name = name[:100] + "..."

    return name + ext


def create_short_temp_path(original_path, temp_dir):
    """
    为长路径文件创建简短的临时路径
    """
    # 生成一个简单的文件名
    import uuid
    temp_name = f"temp_{uuid.uuid4().hex[:8]}.doc"
    temp_path = os.path.join(temp_dir, temp_name)

    return temp_path


def convert_single_doc_optimized(word_app, doc_path, output_path):
    """
    优化版本：转换单个.doc文件为.docx
    word_app: 一个已启动的Word应用程序实例
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(doc_path):
            print(f"   文件不存在: {doc_path}")
            return False

        # 检查文件大小
        file_size = os.path.getsize(doc_path)
        if file_size == 0:
            print(f"   文件为空: {doc_path}")
            return False

        print(f"   正在转换: {os.path.basename(doc_path)}")
        print(f"    文件大小: {file_size / 1024:.1f} KB")

        # 创建一个临时目录来处理长文件名
        with tempfile.TemporaryDirectory() as temp_dir:
            # 为长路径文件创建简短的临时路径
            temp_input_path = create_short_temp_path(doc_path, temp_dir)
            temp_output_path = temp_input_path.replace('.doc', '.docx')

            try:
                # 复制原文件到临时位置（使用简短路径）
                shutil.copy2(doc_path, temp_input_path)
                print(f"    已复制到临时位置: {temp_input_path}")

                # 使用传入的word_app实例打开文件
                print(f"    正在用Word打开文件...")
                doc = word_app.Documents.Open(temp_input_path)

                # 保存为.docx格式
                print(f"    正在保存为.docx...")
                doc.SaveAs2(temp_output_path, FileFormat=16)  # 16 = .docx格式
                doc.Close(SaveChanges=False)

                # 检查输出文件是否存在
                if not os.path.exists(temp_output_path):
                    print(f"   转换失败: 输出文件未创建")
                    return False

                # 复制转换后的文件到目标位置
                shutil.copy2(temp_output_path, output_path)

                print(f"   转换完成")
                return True

            except Exception as e:
                print(f"   Word操作失败: {str(e)}")
                print(f"      详细错误: {traceback.format_exc()}") # 添加详细错误追踪
                return False

    except Exception as e:
        print(f"   转换过程失败: {str(e)}")
        print(f"      详细错误: {traceback.format_exc()}")
        return False


def batch_convert_docs(input_folder, force_overwrite=False):
    """批量转换指定文件夹中的.doc文件"""
    docs_folder = Path(input_folder)
    if not docs_folder.exists():
        print(f" 输入文件夹不存在: {docs_folder}")
        return False

    # 检查win32com库
    win32com_available, pythoncom, wc = check_win32com()
    if not win32com_available:
        return False

    print(f"\n 正在扫描输入文件夹: {docs_folder}")

    # 查找所有.doc文件（统一使用小写扩展名）
    doc_files = [f for f in docs_folder.iterdir() if f.is_file() and f.suffix.lower() == '.doc']

    # 去重（按文件名）
    unique_files = []
    seen_names = set()
    for file_path in doc_files:
        if file_path.name not in seen_names:
            seen_names.add(file_path.name)
            unique_files.append(file_path)

    print(f" 找到 {len(unique_files)} 个.doc文件:")
    for i, doc_file in enumerate(unique_files, 1):
        size_kb = doc_file.stat().st_size / 1024
        print(f"  {i}. {doc_file.name} ({size_kb:.1f} KB)")

    if not unique_files:
        print(" 未找到.doc文件")
        return True

    # 创建backup文件夹
    backup_folder = docs_folder / "backup"
    backup_folder.mkdir(exist_ok=True)
    print(f" 备份文件夹: {backup_folder}")

    # --- 优化部分开始 ---
    # 1. 初始化 COM 并启动 Word 应用
    print(" 启动Word引擎...")
    pythoncom.CoInitialize()
    word_app = None
    try:
        word_app = wc.Dispatch("Word.Application")
        word_app.Visible = False
        word_app.DisplayAlerts = False
        print(" Word引擎已就绪")
    except Exception as e:
        print(f" 无法启动Word应用程序: {e}")
        pythoncom.CoUninitialize()
        return False
    # --- 优化部分结束 ---

    # 开始转换
    print(" 开始批量转换...")

    success_count = 0
    failed_files = []

    for i, doc_file in enumerate(unique_files, 1):
        print(f"\n[{i}/{len(unique_files)}] 处理: {doc_file.name}")

        # 检查输出文件是否已存在
        output_path = doc_file.with_suffix('.docx')
        if output_path.exists() and not force_overwrite:
            print(f"    目标文件已存在，跳过: {output_path.name}")
            success_count += 1
            continue
        elif output_path.exists() and force_overwrite:
            print(f"   目标文件已存在，--force 参数启用，正在覆盖: {output_path.name}")

        # 2. 将 word_app 实例传递给转换函数
        success = convert_single_doc_optimized(word_app, str(doc_file), str(output_path))

        if success:
            # 备份原始文件到backup文件夹
            backup_path = backup_folder / doc_file.name
            try:
                if doc_file.exists():
                    shutil.copy2(doc_file, backup_path)
                    print(f"   已备份到backup文件夹")

                # 删除原始.doc文件
                try:
                    doc_file.unlink()
                    print(f"    已删除原始文件")
                except Exception as e:
                    print(f"    无法删除原始文件: {e}")

                success_count += 1

            except Exception as e:
                print(f"    转换成功但备份/清理失败: {e}")
                success_count += 1
        else:
            failed_files.append(f"{doc_file.name}")

    # --- 优化部分开始 ---
    # 3. 在所有转换完成后，统一退出 Word 应用并反初始化 COM
    print("\n 正在关闭Word引擎...")
    if word_app:
        try:
            word_app.Quit()
            print(" Word引擎已关闭")
        except Exception as e:
            print(f"  关闭Word时出错: {e}")
    try:
        pythoncom.CoUninitialize()
        print(" COM 组件已清理")
    except Exception as e:
        print(f"  清理COM时出错: {e}")
    # --- 优化部分结束 ---

    # 显示统计信息
    print("\n" + "="*50)
    print(" 转换统计")
    print("="*50)
    print(f" 成功转换: {success_count}/{len(unique_files)} 个文件")

    if failed_files:
        print(f" 转换失败: {len(failed_files)} 个文件")
        for filename in failed_files:
            print(f"  - {filename}")

        print("\n 对于转换失败的文件，可以尝试:")
        print("  1. 手动用Microsoft Word打开并另存为.docx格式")
        print("  2. 确保文件没有损坏")
        print("  3. 缩短文件名（建议少于50字符）")
    else:
        print(" 所有文件转换成功!")

    # 检查转换结果
    print("\n 转换结果检查:")

    converted_files = list(docs_folder.glob("*.docx"))
    remaining_doc_files = list(docs_folder.glob("*.doc"))

    print(f"   {docs_folder}:")
    print(f"    - .docx文件: {len(converted_files)} 个")
    print(f"    - .doc文件: {len(remaining_doc_files)} 个 (不含backup)")

    backup_files = list(backup_folder.glob("*.doc"))
    print(f"   {backup_folder}:")
    print(f"    - .doc文件备份: {len(backup_files)} 个")

    return success_count > 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量转换.doc文件为.docx格式")
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='./docs',
        help='输入文件夹路径 (默认: ./docs)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制覆盖已存在的.docx文件'
    )
    args = parser.parse_args()

    # 检查并请求管理员权限
    if not is_admin():
        request_admin_and_restart()

    print("="*60)
    print(" .doc 文件批量转换工具")
    print("="*60)

    # 显示当前文件夹结构
    docs_folder = Path(args.input)
    if docs_folder.exists():
        print(f"\n当前输入文件夹 '{docs_folder}' 内容:")
        doc_count = 0
        docx_count = 0
        for item in docs_folder.iterdir():
            if item.is_file():
                if item.suffix.lower() == '.doc':
                    print(f"   {item.name}")
                    doc_count += 1
                elif item.suffix.lower() == '.docx':
                    print(f"   {item.name} (已转换)")
                    docx_count += 1

        if doc_count == 0:
            print("   没有发现.doc文件，无需转换")
            if docx_count > 0:
                print(f"   已有 {docx_count} 个.docx文件。")
            input("\n按回车键退出...")
            return True
    else:
        print(f" 输入文件夹不存在: {docs_folder}")
        input("\n按回车键退出...")
        return False

    try:
        # 执行批量转换
        start_time = time.time()
        print(f"\n开始自动转换...")
        if args.force:
            print(f"注意: --force 模式已启用，将覆盖已存在的 .docx 文件。")
        success = batch_convert_docs(args.input, args.force)
        end_time = time.time()

        print(f"\n  转换耗时: {end_time - start_time:.2f}秒")

        if success:
            print("\n 转换完成！现在可以正常运行主程序")
        else:
            print("\n  转换过程中出现错误，请检查上方日志")

    except KeyboardInterrupt:
        print("\n\n  用户中断操作")
        success = False
    except Exception as e:
        print(f"\n 发生错误: {str(e)}")
        print("\n详细错误信息:")
        traceback.print_exc()
        success = False

    input("\n按回车键退出...")
    return success


if __name__ == "__main__":
    main()