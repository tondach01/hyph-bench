import argparse
import os
import subprocess
import shutil

THICK_SPACER = "=" * 50
THIN_SPACER = "-" * 50
TMP_PATH = "/tmp/hyph-bench"

def implement_fixes(wordlist: str, fixes: str):
    subprocess.call(f"python -m scripts.replace_in_wordlist --wordlist {wordlist} --fixed {fixes} --output-dir {TMP_PATH}", shell=True)
    subprocess.call(f"mv -f {TMP_PATH}/fixed.wlh {wordlist}", shell=True)

def compare_fixes(paths: list[str], result_dir: str):
    print(THIN_SPACER)
    if len(paths) == 1:
        print("Only one file provided, no comparision will be done")

    print("Comparing between provided files")
    args = " ".join(paths)
    result_path = os.path.join(result_dir, "comparision.csv")
    subprocess.call(f"python -m scripts.compare_annotations {args} > {result_path}", shell=True)
    print(f"Comparision written to {result_path}")
    print(THIN_SPACER)

def run_optimizer(params: str):
    print("\nRunning optimizer")
    print(THIN_SPACER)
    subprocess.call("python -m scripts.optimize " + params, shell=True)

def get_fix_files(last_order: list[str] = []) -> list[str]:
    print("\nOptimizer output a file to its results dir with words which were incorrectly hyphenated")
    print(THIN_SPACER)
    
    if last_order:
        print("Last time provided in this order")
    for i, file in enumerate(last_order):
        print(f"{i + 1}. {file}")

    repeat = True
    while repeat:
        response = input("Please re-hyphenate the words and input paths to the results in the same order as last time or press enter to stop:\n")
        paths = response.strip().split()

        repeat = False
        print("Following files provided:")
        for i, file in enumerate(paths):
            if not os.path.exists(file):
                print(f"Path {file} does not exist. Please check it and re-enter the paths")
                repeat = True
            else:
                print(f"{i + 1}. {file}")

    return paths

def create_iter_folder(n: int, output: str) -> str:
    iteration_output = os.path.join(output, f"iter{n}")
    os.makedirs(iteration_output)
    return iteration_output

def preprocess_params(params: list[str], wordlist: str, translate: str):
    try:
        wl_param_i = params.index("--wordlist") + 1
    except ValueError:
        params.append("--wordlist")
        params.append(wordlist)
        wl_param_i = len(params) - 1

    try:
        translate_param_i = params.index("--translate") + 1
        params[translate_param_i] = translate
    except ValueError:
        params.append("--translate")
        params.append(translate)

    try:
        optimizer_output_dir = params[params.index("--output-dir") + 1]
    except ValueError:
        optimizer_output_dir = "results"

    def modifer(wordlist_path):
        params[wl_param_i] = wordlist_path

    params.append("--export-iteration-results")

    return modifer, optimizer_output_dir, params[params.index("--lang") + 1]

def save_optimizer_results(wl_index: int, output_dir: str, result_dir: str, lang: str):
    bad_path = os.path.join(output_dir, f"{wl_index}bad.txt")
    optimizer_csv_path = os.path.join(output_dir, f"{wl_index}optimizer_result.csv")
    patterns_path = os.path.join(output_dir, f"{wl_index}_{lang}.pat")

    result_bad_path = os.path.join(result_dir, f"{lang}_bad.txt")
    result_optimizer_csv_path = os.path.join(result_dir, f"{lang}_history.csv")
    result_patterns_path = os.path.join(result_dir, f"{lang}_final.pat")
    
    shutil.move(result_bad_path, bad_path)
    shutil.move(result_optimizer_csv_path, optimizer_csv_path)
    shutil.move(result_patterns_path, patterns_path)

def main():
    parser = argparse.ArgumentParser(
        description="Iterate over provided wordlist for n iternations or until convergence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--input-wordlist", required=True, type=str,
                        help="Hyphenated input wordlist (one word per line)")
    parser.add_argument("--input-translate", required=True, type=str,
                        help="Translate file for the wordlist")
    parser.add_argument("--output-dir", required=False, type=str, default="iter_results",
                        help="Output directory path")
    parser.add_argument("--optimizer-params", required=False, type=str, default="--objective bounded_bad --lang uk",
                        help="Params which will be passed to the optimizer; see python -m scripts.optimize --help\n" \
                             "They must point to the same wordlist as the input param, the results folder should be left to default\n" \
                             "Default params: --objective bounded_bad --lang uk")

    args = parser.parse_args()
    
    output = args.output_dir
    params = args.optimizer_params.split()
    wl_name = os.path.basename(args.input_wordlist)
    wordlist = os.path.join(TMP_PATH, wl_name)
    iter_folders = os.path.join(output, "iter*")
    prev_wordlists = os.path.join(output, "*.wlh")
    add_wordlist_to_params, optimizer_output, lang \
        = preprocess_params(params, wordlist, args.input_translate)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(TMP_PATH, exist_ok=True)

    shutil.copy(args.input_wordlist, wordlist)

    if (os.path.exists(output)):
        subprocess.call(f"rm -rf {iter_folders}", shell=True)
        subprocess.call(f"rm {prev_wordlists}", shell=True)

    print(THICK_SPACER)
    print(f"Iterating on {args.input_wordlist}")
    print(f"Results will be stored to {output}")
    print("Once there are few enough words in the bad.txt file output by the optimizer, we can stop")
    print(THICK_SPACER + "\n")

    # inital run to get 500
    n = 0
    print(f"ITERATION {n}")
    print(THIN_SPACER)

    iteration_output = create_iter_folder(n, output)
    run_optimizer(" ".join(params))

    shutil.copy(os.path.join(optimizer_output, f"{lang}_bad.txt"), iteration_output)

    copied_wordlists = []
    fix_files = get_fix_files()
    compare_fixes(fix_files, iteration_output)

    for i, fix in enumerate(fix_files):
        copy_path = os.path.join(TMP_PATH, f"{i}.wlh")
        copied_wordlists.append(copy_path)
        shutil.copy(wordlist, copy_path)
        implement_fixes(copy_path, fix)

    while True:
        n += 1
        print(f"\nITERATION {n}")
        print(THIN_SPACER)
        iteration_output = create_iter_folder(n, output)

        for i, wl in enumerate(copied_wordlists):
            add_wordlist_to_params(wl)
            run_optimizer(" ".join(params))
            save_optimizer_results(i, iteration_output, optimizer_output, lang)

        fixes = get_fix_files(fix_files)
        if not fixes:
            break

        for i, file in enumerate(fixes):
            implement_fixes(f"{TMP_PATH}/{i}.wlh", file, i)

    print(THICK_SPACER)
    print(f"Final iterations {n}")
    print(f"Storing result to {output}")
    print(THICK_SPACER)

    for i, file in enumerate(fix_files):
        name = os.path.basename(file)
        result_path = os.path.join(output, f"{i}{name}.wlh")
        shutil.move(f"{TMP_PATH}/{i}.wlh", result_path)


if __name__ == "__main__":
    main()