#!/bin/bash

# 主控脚本 - 依次运行所有汉明码测试

echo "=== 开始运行所有汉明码测试 ==="
echo "开始时间: $(date)"

# 运行4位汉明码测试
echo -e "\n=== 运行4位汉明码测试 ==="
./run_ARGH_4bit.sh

# 检查上一个命令是否成功执行
if [ $? -eq 0 ]; then
    echo "4位汉明码测试完成"
else
    echo "4位汉明码测试失败，但继续执行下一个测试"
fi

# 运行8位汉明码测试
echo -e "\n=== 运行8位汉明码测试 ==="
./run_ARGH_8bit.sh

if [ $? -eq 0 ]; then
    echo "8位汉明码测试完成"
else
    echo "8位汉明码测试失败，但继续执行下一个测试"
fi

# 运行16位汉明码测试
echo -e "\n=== 运行16位汉明码测试 ==="
./run_ARGH_16bit.sh

if [ $? -eq 0 ]; then
    echo "16位汉明码测试完成"
else
    echo "16位汉明码测试失败，但继续执行下一个测试"
fi

# 运行32位汉明码测试
echo -e "\n=== 运行32位汉明码测试 ==="
./run_ARGH_32bit.sh

if [ $? -eq 0 ]; then
    echo "32位汉明码测试完成"
else
    echo "32位汉明码测试失败"
fi

echo -e "\n=== 所有测试完成 ==="
echo "结束时间: $(date)"