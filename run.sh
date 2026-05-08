cd /home/gg/hermes-agent-simple

# 首次安装，推荐
uv sync --all-extras
source .venv/bin/activate

/home/gg/.hermes/config.yaml
/home/gg/.hermes/.env


# 初始化配置 / API key
./hermes setup

# 检查环境
./hermes doctor

# 交互式聊天
./hermes

# 单次调用
./hermes -z "1+1"

# TUI
./hermes --tui

# 查看模型配置
./hermes model

# 管理 MCP client
./hermes mcp list
./hermes mcp add ...
./hermes mcp test ...

# gateway，仅保留飞书 / 企业微信相关能力
./hermes gateway setup
./hermes gateway start
./hermes gateway status
