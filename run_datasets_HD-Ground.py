import os
import yaml
import subprocess

def find_file_with_prefix(directory, prefix):
    for file in os.listdir(directory):
        if file.startswith(prefix):
            return os.path.join(directory, file)
    return None

# 配置文件路径
config_file_path = 'src/kcc_slam/configs/config_HD.yaml'

# 数据集存放的根目录
dataset_root = '/media/zheng/My_Passport/home/zheng/datasets/ground-texture/HD_ground'

# 结果存放的根目录
saving_root = '/home/zheng/projects/kcc_slam/src/kcc_slam/saving'

# 获取数据集目录下所有的文件夹
dataset_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

kernel_types = ['polynomial', 'guassian']
for dataset_dir in dataset_dirs:
    for kernel_type in kernel_types:
        
        cur_saving_dir = os.path.join(saving_root, f'HD_Ground_{kernel_type}')
        have_done_datasets = os.listdir(cur_saving_dir)
        
        # 提取数据集名称
        dataset_name = os.path.basename(dataset_dir)
        seqs = os.listdir(os.path.join(dataset_dir, 'database'))
        for seq in seqs:
            if ("HD_"+dataset_name+"_database_"+seq) in have_done_datasets:
                print("%s will not be performed" % dataset_name)
                continue
            else:
                print("processing %s, its saving root is %s, its kernel type is %s" % (dataset_dir,saving_root, kernel_type))
            # 读取配置文件
            with open(config_file_path, 'r') as config_file:
                config = yaml.safe_load(config_file)

            # 更新配置文件
            config['dataset']['dataroot'] = os.path.join(dataset_dir, 'database', seq)
            config['saving']['saving_root'] = os.path.join(saving_root, f'HD_Ground_{kernel_type}', f'HD_{dataset_name}_database_{seq}')
            if kernel_type == "polynomial":
                config['correlation_flow']['kernel'] = 0
            elif kernel_type == "guassian":
                config['correlation_flow']['kernel'] = 1
            else:
                print("wrong kernel type")
                break


            # 将修改后的配置文件保存
            with open(config_file_path, 'w') as config_file:
                yaml.safe_dump(config, config_file)

            # 运行程序
            cmd = f'rosrun kcc_slam kcc_slam {config_file_path}'
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError:
                print(f"Error: Failed to run dataset {dataset_name}")
                continue

            # 查找结果文件和真值文件
            result_file = find_file_with_prefix(config['saving']['saving_root'], "KCC_Keyframe")
            ground_truth_file = find_file_with_prefix(config['dataset']['dataroot'], "groundtruth")

            if result_file is not None and ground_truth_file is not None:
                # 运行evo_ape评估
                eval_cmd1 = f"evo_ape tum {ground_truth_file} {result_file} -as"
                eval_cmd2 = f"evo_traj tum --ref {ground_truth_file} {result_file} -as"
                output_file_path1 = os.path.join(config['saving']['saving_root'], "result.txt")
                output_file_path2 = os.path.join(config['saving']['saving_root'], "length.txt")

                # 将评估结果重定向到result.txt文件
                with open(output_file_path1, 'w') as output_file1:
                    subprocess.run(eval_cmd1, shell=True, stdout=output_file1)
                # 将评估结果重定向到result.txt文件
                with open(output_file_path2, 'w') as output_file2:
                    subprocess.run(eval_cmd2, shell=True, stdout=output_file2)
            else:
                print(f"Error: Cannot find result or ground truth file for dataset {dataset_name}")
