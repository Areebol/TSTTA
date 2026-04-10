#!/bin/bash

# 1. 强制切换到指定的工作目录
WORK_DIR="/sharedata/wyc/code/TSTTA_Codebook"
cd "${WORK_DIR}" || { echo "❌ 切换到工作目录 ${WORK_DIR} 失败，请检查路径！"; exit 1; }

echo "当前工作目录: $(pwd)"

# 2. 定义脚本所在目录 (相对路径)
SCRIPT_DIR="./scripts/0401/ablation"

# 3. 定义要串行执行的8个脚本（可以按照你喜欢的逻辑顺序排列）
SCRIPTS=(
    # "Adapter_ETT.sh"
    # "Adapter_eVED.sh"
    # "pretrain_ETT.sh"
    # "pretrain_eVED.sh"
    "Offline_ETT.sh"
    # "Offline_eVED.sh"
    # "full_ETT.sh"
    # "full_eVED.sh"
)

# 4. 开始线性执行
echo "====================================================="
echo "🚀 start running  (total ${#SCRIPTS[@]} scripts)"
echo "====================================================="

for script in "${SCRIPTS[@]}"; do
    script_path="${SCRIPT_DIR}/${script}"
    
    # 检查脚本文件是否存在
    if [ ! -f "$script_path" ]; then
        echo "⚠️ can not find : $script_path，skip..."
        continue
    fi
    
    # 赋予执行权限 (防止新建的脚本没有+x)
    chmod +x "$script_path"
    
    echo "-----------------------------------------------------"
    echo "▶️  running: ${script}"
    echo "⏰ strat time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "-----------------------------------------------------"
    
    # 执行具体的脚本
    bash "$script_path"
    
    # 获取脚本执行的返回状态码
    status=$?
    
    if [ $status -eq 0 ]; then
        echo "✅ ${script} finished successfully!"
    else
        echo "❌ ${script} execution was interrupted or an error occurred (Exit Code: $status)."
        echo "⚠️ An exception occurred in the script, halting subsequent experiments to prevent cascading errors."
        exit 1  # <--- If you want the script to continue running even if one fails, comment out this line
    fi
    
    echo -e "⏰ 结束时间: $(date '+%Y-%m-%d %H:%M:%S')\n\n"
done

echo "====================================================="
echo "🎉 All scripts executed successfully!"
echo "====================================================="