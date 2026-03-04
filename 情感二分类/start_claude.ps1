# Claude CLI 配置脚本
# 设置 Anthropic API 密钥
$env:ANTHROPIC_API_KEY="sk-691331534d4a403fbd2add1841357a8f"

# 设置 Anthropic API 基础 URL (本地代理)
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8045"

# 显示配置信息
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Claude CLI 环境配置" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "API Key: $($env:ANTHROPIC_API_KEY.Substring(0,10))..." -ForegroundColor Cyan
Write-Host "Base URL: $env:ANTHROPIC_BASE_URL" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""

# 启动 Claude CLI
Write-Host "正在启动 Claude CLI..." -ForegroundColor Yellow
claude
