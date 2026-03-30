"""
Document conversion helpers for turning legacy .doc files into .docx files.

This module keeps the standalone CLI utility, and also exposes importable
helpers that can be safely used by the Web knowledge-base build flow.
"""

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List


DOC_FILE_FORMAT = 16
BACKUP_DIR_NAME = "backup"


def is_admin() -> bool:
    """Check whether the current process is running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def request_admin_and_restart() -> None:
    """Restart the current script with admin privileges."""
    script_path = os.path.abspath(sys.argv[0])
    work_dir = os.path.dirname(script_path)

    if script_path.endswith(".py"):
        executable = sys.executable
        params = script_path + " " + " ".join(sys.argv[1:])
    else:
        executable = script_path
        params = " ".join(sys.argv[1:])

    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        executable,
        params,
        work_dir,
        1,
    )

    if result <= 32:
        raise RuntimeError("Unable to request admin privileges and restart the program.")

    sys.exit(0)


def check_win32com(allow_install: bool = True):
    """Load pywin32 dependencies, optionally attempting installation."""
    try:
        import pythoncom
        from win32com import client as wc

        return True, pythoncom, wc, ""
    except ImportError as exc:
        if not allow_install:
            message = (
                "Missing pywin32; .doc preprocessing will be skipped in this run. "
                "Install it manually with: pip install pywin32"
            )
            return False, None, None, f"{message} ({exc})"

        print("Missing required dependency: pywin32")
        print("Trying to install pywin32...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pywin32"])

            import pythoncom
            from win32com import client as wc

            print("pywin32 installed successfully")
            return True, pythoncom, wc, ""
        except Exception as install_exc:
            print(f"Failed to install pywin32: {install_exc}")
            print("Please run manually: pip install pywin32")
            return False, None, None, str(install_exc)


def create_short_temp_path(temp_dir: str, suffix: str = ".doc") -> str:
    """Create a short temporary path to avoid long-path issues in Word."""
    import uuid

    temp_name = f"temp_{uuid.uuid4().hex[:8]}{suffix}"
    return os.path.join(temp_dir, temp_name)


def _close_word_document(doc) -> None:
    """Close a Word document if it is open."""
    if doc is None:
        return
    try:
        doc.Close(SaveChanges=False)
    except Exception:
        pass


def _is_relative_backup_path(relative_path: Path, backup_dir_name: str = BACKUP_DIR_NAME) -> bool:
    return any(part.lower() == backup_dir_name.lower() for part in relative_path.parts)


def iter_doc_files(
    input_folder: Path,
    recursive: bool = True,
    skip_backup: bool = True,
    backup_dir_name: str = BACKUP_DIR_NAME,
) -> List[Path]:
    """Discover .doc files under a folder."""
    root = Path(input_folder)
    candidates: Iterable[Path]
    if recursive:
        candidates = root.rglob("*.doc")
    else:
        candidates = root.glob("*.doc")

    doc_files: List[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            relative_path = path.relative_to(root)
        except ValueError:
            relative_path = path
        if skip_backup and _is_relative_backup_path(relative_path, backup_dir_name):
            continue
        if path.suffix.lower() != ".doc":
            continue
        doc_files.append(path)

    return sorted(doc_files, key=lambda item: str(item.relative_to(root)).lower())


def build_backup_path(
    root_folder: Path,
    doc_path: Path,
    backup_dir_name: str = BACKUP_DIR_NAME,
) -> Path:
    """Build a backup destination preserving the original relative path."""
    relative_path = doc_path.relative_to(root_folder)
    return root_folder / backup_dir_name / relative_path


def ensure_unique_path(path: Path) -> Path:
    """Avoid overwriting an existing backup file."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def convert_single_doc_optimized(word_app, doc_path: str, output_path: str) -> bool:
    """
    Convert a single .doc file to .docx using an existing Word application.
    """
    try:
        if not os.path.exists(doc_path):
            print(f"   File does not exist: {doc_path}")
            return False

        file_size = os.path.getsize(doc_path)
        if file_size == 0:
            print(f"   File is empty: {doc_path}")
            return False

        print(f"   Converting: {os.path.basename(doc_path)}")
        print(f"    File size: {file_size / 1024:.1f} KB")

        with tempfile.TemporaryDirectory() as temp_dir:
            doc = None
            temp_input_path = create_short_temp_path(temp_dir, suffix=".doc")
            temp_output_path = temp_input_path.replace(".doc", ".docx")

            try:
                shutil.copy2(doc_path, temp_input_path)
                print(f"    Copied to temporary path: {temp_input_path}")

                print("    Opening in Word...")
                doc = word_app.Documents.Open(temp_input_path)

                print("    Saving as .docx...")
                doc.SaveAs2(temp_output_path, FileFormat=DOC_FILE_FORMAT)
                _close_word_document(doc)
                doc = None

                if not os.path.exists(temp_output_path):
                    print("   Conversion failed: output .docx was not created")
                    return False

                shutil.copy2(temp_output_path, output_path)
                print("   Conversion completed")
                return True

            except Exception as exc:
                print(f"   Word conversion failed: {exc}")
                print(f"      Traceback: {traceback.format_exc()}")
                return False
            finally:
                _close_word_document(doc)

    except Exception as exc:
        print(f"   Conversion process failed: {exc}")
        print(f"      Traceback: {traceback.format_exc()}")
        return False


def _empty_preprocess_report(source_dir: Path) -> Dict[str, object]:
    return {
        "source_dir": str(source_dir),
        "detected_doc_files": 0,
        "converted_doc_files": 0,
        "archived_doc_files": 0,
        "skipped_existing_docx": [],
        "failed_doc_files": [],
        "archive_failed_doc_files": [],
        "errors": [],
    }


def preprocess_doc_files_for_build(
    input_folder,
    force_overwrite: bool = False,
    skip_backup: bool = True,
    backup_dir_name: str = BACKUP_DIR_NAME,
    allow_install: bool = False,
) -> Dict[str, object]:
    """
    Recursively preprocess .doc files before a knowledge-base build.

    Behavior:
    - Scan for .doc files recursively
    - Skip anything inside backup/
    - Convert to same-folder .docx
    - Move original .doc into backup/<relative path>/
    - Continue gracefully on failures
    """
    source_dir = Path(input_folder)
    report = _empty_preprocess_report(source_dir)

    if not source_dir.exists() or not source_dir.is_dir():
        report["errors"].append(f"Source directory does not exist or is not a folder: {source_dir}")
        return report

    doc_files = iter_doc_files(
        source_dir,
        recursive=True,
        skip_backup=skip_backup,
        backup_dir_name=backup_dir_name,
    )
    report["detected_doc_files"] = len(doc_files)

    if not doc_files:
        return report

    available, pythoncom, wc, dependency_message = check_win32com(allow_install=allow_install)
    if not available:
        report["errors"].append(
            dependency_message or "Unable to initialize pywin32 / Word support for .doc preprocessing."
        )
        report["failed_doc_files"] = [str(path.relative_to(source_dir)) for path in doc_files]
        return report

    print(f"Found {len(doc_files)} .doc file(s) that need preprocessing.")
    pythoncom.CoInitialize()
    word_app = None
    try:
        try:
            word_app = wc.Dispatch("Word.Application")
            word_app.Visible = False
            word_app.DisplayAlerts = False
        except Exception as exc:
            report["errors"].append(f"Unable to start Microsoft Word for .doc preprocessing: {exc}")
            report["failed_doc_files"] = [str(path.relative_to(source_dir)) for path in doc_files]
            return report

        for index, doc_file in enumerate(doc_files, start=1):
            relative_path = doc_file.relative_to(source_dir)
            print(f"[{index}/{len(doc_files)}] Preprocessing .doc: {relative_path}")
            output_path = doc_file.with_suffix(".docx")

            if output_path.exists() and not force_overwrite:
                message = f"{relative_path} (same-name .docx already exists)"
                print(f"   Skipped: {message}")
                report["skipped_existing_docx"].append(message)
                continue

            if convert_single_doc_optimized(word_app, str(doc_file), str(output_path)):
                report["converted_doc_files"] += 1
                backup_path = ensure_unique_path(
                    build_backup_path(source_dir, doc_file, backup_dir_name=backup_dir_name)
                )
                try:
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(doc_file), str(backup_path))
                    report["archived_doc_files"] += 1
                    print(f"   Archived original .doc to: {backup_path.relative_to(source_dir)}")
                except Exception as exc:
                    report["archive_failed_doc_files"].append(
                        f"{relative_path} -> {backup_path.relative_to(source_dir)} ({exc})"
                    )
                    print(f"   Converted to .docx, but failed to move original .doc: {exc}")
            else:
                report["failed_doc_files"].append(str(relative_path))

    finally:
        if word_app is not None:
            try:
                word_app.Quit()
            except Exception:
                pass
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    return report


def batch_convert_docs(input_folder, force_overwrite: bool = False) -> bool:
    """
    Standalone CLI behavior: recursively convert .doc files in the folder.
    """
    docs_folder = Path(input_folder)
    if not docs_folder.exists():
        print(f"Input folder does not exist: {docs_folder}")
        return False

    available, pythoncom, wc, dependency_message = check_win32com(allow_install=True)
    if not available:
        print(dependency_message)
        return False

    doc_files = iter_doc_files(
        docs_folder,
        recursive=True,
        skip_backup=True,
        backup_dir_name=BACKUP_DIR_NAME,
    )
    if not doc_files:
        print("No .doc files found")
        return True

    backup_folder = docs_folder / BACKUP_DIR_NAME
    backup_folder.mkdir(exist_ok=True)

    pythoncom.CoInitialize()
    word_app = None
    success_count = 0
    failed_files: List[str] = []
    try:
        word_app = wc.Dispatch("Word.Application")
        word_app.Visible = False
        word_app.DisplayAlerts = False

        for index, doc_file in enumerate(doc_files, start=1):
            relative_path = doc_file.relative_to(docs_folder)
            print(f"\n[{index}/{len(doc_files)}] Processing: {relative_path}")
            output_path = doc_file.with_suffix(".docx")
            if output_path.exists() and not force_overwrite:
                print(f"   Target .docx already exists, skipping: {output_path.relative_to(docs_folder)}")
                success_count += 1
                continue

            if convert_single_doc_optimized(word_app, str(doc_file), str(output_path)):
                backup_path = ensure_unique_path(
                    build_backup_path(docs_folder, doc_file, backup_dir_name=BACKUP_DIR_NAME)
                )
                try:
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(doc_file), str(backup_path))
                    print(f"   Archived original .doc to: {backup_path.relative_to(docs_folder)}")
                except Exception as exc:
                    print(f"   Converted to .docx, but failed to move original .doc: {exc}")
                success_count += 1
            else:
                failed_files.append(str(relative_path))
    except Exception as exc:
        print(f"Unable to start Microsoft Word: {exc}")
        return False
    finally:
        if word_app is not None:
            try:
                word_app.Quit()
            except Exception:
                pass
        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    print("\nConversion summary")
    print(f"Successful conversions: {success_count}/{len(doc_files)}")
    if failed_files:
        print(f"Failed conversions: {len(failed_files)}")
        for filename in failed_files:
            print(f" - {filename}")
    else:
        print("All .doc files were handled successfully.")
    return success_count > 0


def main() -> bool:
    parser = argparse.ArgumentParser(description="Batch convert .doc files to .docx")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./docs",
        help="Input folder path (default: ./docs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing same-name .docx file",
    )
    args = parser.parse_args()

    if not is_admin():
        request_admin_and_restart()

    print("=" * 60)
    print(".doc batch conversion tool")
    print("=" * 60)

    docs_folder = Path(args.input)
    if not docs_folder.exists():
        print(f"Input folder does not exist: {docs_folder}")
        input("\nPress Enter to exit...")
        return False

    print(f"\nCurrent input folder: {docs_folder}")
    recursive_docs = iter_doc_files(docs_folder, recursive=True, skip_backup=True)
    if not recursive_docs:
        print("No .doc files were found in the folder or its subfolders. Nothing to convert.")
        input("\nPress Enter to exit...")
        return True

    print(f"Found {len(recursive_docs)} .doc file(s) in the folder tree.")

    try:
        start_time = time.time()
        success = batch_convert_docs(args.input, args.force)
        elapsed = time.time() - start_time
        print(f"\nElapsed time: {elapsed:.2f}s")
        if success:
            print("\nConversion finished. You can now run the main program.")
        else:
            print("\nConversion completed with errors. Please review the log above.")
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
        success = False
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        traceback.print_exc()
        success = False

    input("\nPress Enter to exit...")
    return success


if __name__ == "__main__":
    main()
