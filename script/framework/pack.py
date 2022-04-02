#!/usr/bin/env python3
# coding: utf-8

import os
import re
import sys
import shutil
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Using lipo to pack frameworks to one')
    parser.add_argument('framework', type=str,
                        help='path contains several frameworks')
    parser.add_argument('output', type=str, default=".", nargs='?',
                        help='path to write output framework')
    args = parser.parse_args()
    return args


regex_framework = re.compile(r"^(.*)\.framework$", re.I)


def list_frameworks(root):
    """
    :return: {name, directories[string]}
    """
    found = {}
    for local, dirs, files in os.walk(root):
        for d in dirs:
            match = re.match(regex_framework, d)
            if match:
                name = match[1]
                path = os.path.join(local, d)
                if name in found:
                    found[name].append(path)
                else:
                    found[name] = [path, ]
    return found


SPACE = 60


def start_select(selections):
    print("=" * SPACE)

    i = 0
    for selection in selections:
        i += 1
        number = "{}: ".format(i)
        space = " " * len(number)
        selection = selection.replace("\n", "\n" + space)
        print(number + selection)


def do_select(selections, tip, default=None):
    print("-" * SPACE)
    N = len(selections)

    if default is None:
        default = " ".join(map(str, range(1, N + 1)))

    chosen_set = re.findall(r"\d+", default)
    chosen_set = list(map(int, chosen_set))

    chosen = input("{} (default {}): ".format(tip, chosen_set))

    chosen = chosen.strip()
    if not chosen:
        chosen = default
    chosen = re.findall(r"\d+", chosen)
    split = []
    for num in chosen:
        # check each number if in [1, N]
        i = int(num)
        if i < 1:
            continue
        if i <= N:
            split.append(i)
            continue
        # split number, like 456 to [4, 5, 6]
        while num:
            for n in range(1, N + 1):
                n += 1
                if n > N:
                    split.append(int(num))
                    num = ''
                    break
                elif int(num[:n]) > N:
                    n -= 1
                    split.append(int(num[:n]))
                    num = num[n:]
                    break
    chosen = []
    for i in split:
        if 0 < i <= N and i not in chosen:
            chosen.append(i)
    print("Chosen:", chosen)
    return chosen


def finish_select():
    print("=" * SPACE)


def lipo_info(path):
    """
    Input path got arch info
    :param path:
    :return:
    """
    with os.popen("lipo -info \"{}\"".format(path)) as result:
        arch = result.read()
        arch = arch.strip()
        arch = arch.replace(path + " ", "")
        colon = arch.rfind(":")
        if colon >= 0:
            arch = arch[colon + 1:].strip()
        return arch


def lipo_create(create, output):
    cmd = "lipo -create {} -output {}".format(
        " ".join(create), output)
    with os.popen(cmd) as result:
        return result.read()


def do_folder_copy(src, dst):
    """
    copy(a, b) result is got `a/b` folder
    :param src:
    :param dst:
    :return:
    """
    source = src
    output = os.path.join(dst, os.path.split(src)[-1])
    if not os.path.isdir(output):
        os.makedirs(output)
    shutil.copytree(source, output, dirs_exist_ok=True)


def main():
    args = parse()

    framework_root = args.framework
    output_path = args.output

    # 1. List all frameworks

    frameworks = list_frameworks(framework_root)

    if len(frameworks) <= 0:
        sys.stderr.write("[ERROR] Can not found any frameworks in {}\n".format(framework_root))
        exit()

    # 2. Choose the framework to pack

    framework_names = sorted(frameworks.keys())
    if len(frameworks) == 1:
        target = framework_names[0]
    else:
        start_select(framework_names)
        while True:
            selection = do_select(framework_names, "Choose one framework to pack", "1")
            if len(selection) != 1:
                sys.stderr.write("[ERROR] Must choose one framework, but got {}\n".format(selection))
                continue
            break
        finish_select()
        target = framework_names[selection[0] - 1]
    print("[INFO] Select framework {}".format(target))

    # 3. Choose arch to pack

    using_frameworks = sorted(frameworks[target])
    using_lipo_info = []
    for path in using_frameworks:
        selection_title = "[{}]".format(path)
        info = lipo_info(os.path.join(path, target))
        using_lipo_info.append(" ".join([selection_title, info]))

    start_select(using_lipo_info)
    while True:
        selection = do_select(using_lipo_info, "Choose frameworks to pack")
        if len(selection) < 1:
            sys.stderr.write("[ERROR] Must chose at least one framework, but got {}\n".format(selection))
            continue
        break
    while True:
        template = do_select(using_lipo_info, "Choose one framework to be template", "{}".format(selection[0]))
        if len(template) != 1:
            sys.stderr.write("[ERROR] Must choose one template, but got {}\n".format(template))
            continue
        break
    finish_select()

    # 4. do pack
    framework_name = target
    template_framework = using_frameworks[template[0] - 1]
    chosen_frameworks = [using_frameworks[i - 1] for i in selection]
    output_framework = os.path.join(output_path, os.path.split(template_framework)[-1])
    chosen_libraries = [os.path.join(p, framework_name) for p in chosen_frameworks]
    output_library = os.path.join(output_framework, framework_name)

    # 4.1 check if output path exists
    # done in copy

    # 4.2 do copy firstly
    do_folder_copy(template_framework, output_path)
    if os.path.isfile(output_library):
        os.remove(output_library)

    # 4.3 pack arch
    lipo_create(chosen_libraries, output_library)

    print("[INFO] Output: [{}] {}".format(output_framework, lipo_info(output_library)))


if __name__ == '__main__':
    main()
