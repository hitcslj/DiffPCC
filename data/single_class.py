# 打开文件
with open('/home/guoqing/DiffPC/data/data_split/ShapeNet55-34/ShapeNet-55/test.txt', 'r') as f:
    # 读取文件内容
    content = f.readlines()  

# 打开文件
with open('/home/guoqing/DiffPC/data/data_split/ShapeNet55-34/ShapeNet-55/airplane/test.txt', 'w') as f:
    for line in content:
        if '02691156' in line:
            f.write(line)