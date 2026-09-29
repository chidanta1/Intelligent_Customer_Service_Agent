#!/bin/bash
# 智能客服 Agent 本地服务启动脚本
# 所有运行时数据保存在: runtime_data/

set -e

PROJECT_BASE=/home/dhc/projects/resume/Intelligent_Customer_Service_Agent
CONDA_ENV=intelligent-customer-service-agent
RUNTIME_CACHE="$PROJECT_BASE/runtime_data/cache"

echo "=== 启动智能客服 Agent 本地服务 ==="

# 激活 conda 环境
source /home/dhc/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 将科学计算依赖的缓存放进项目运行目录，避免用户缓存目录不可写时启动失败
mkdir -p "$RUNTIME_CACHE/numba" "$RUNTIME_CACHE/matplotlib"
export NUMBA_CACHE_DIR="$RUNTIME_CACHE/numba"
export MPLCONFIGDIR="$RUNTIME_CACHE/matplotlib"

# 清除代理环境变量（避免 ollama/httpx 报错）
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "[1/4] 启动 MySQL..."
nohup mysqld \
  --datadir="$PROJECT_BASE/runtime_data/mysql_data" \
  --socket="$PROJECT_BASE/runtime_data/mysql.sock" \
  --pid-file="$PROJECT_BASE/runtime_data/mysql.pid" \
  --port=3306 \
  --bind-address=127.0.0.1 \
  --skip-networking=0 \
  --mysqlx=0 \
  --log-error="$PROJECT_BASE/runtime_data/mysql.err" > /dev/null 2>&1 &

echo "[2/4] 启动 Redis..."
nohup redis-server \
  --bind 127.0.0.1 \
  --port 6379 \
  --save '' \
  --appendonly no \
  --dir "$PROJECT_BASE/runtime_data" > /dev/null 2>&1 &

echo "[3/4] 启动 Neo4j..."
NEO4J_HOME="$PROJECT_BASE/runtime_data/neo4j_data"
cd "$NEO4J_HOME"
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
NEO4J_HOME="$NEO4J_HOME" \
NEO4J_CONF="$NEO4J_HOME/conf" \
nohup /usr/share/neo4j/bin/neo4j console > "$NEO4J_HOME/logs/console.log" 2>&1 &

echo "等待数据库服务启动..."
sleep 15

echo "[4/4] 启动后端 API..."
cd "$PROJECT_BASE/code/deepseek_agent/llm_backend"
DB_HOST=127.0.0.1 \
DB_PORT=3306 \
DB_USER=root \
DB_PASSWORD='' \
DB_NAME=kefu_agent \
REDIS_HOST=127.0.0.1 \
REDIS_PORT=6379 \
NEO4J_URL=bolt://127.0.0.1:7687 \
NEO4J_USERNAME=neo4j \
NEO4J_DATABASE=neo4j \
GRAPHRAG_PROJECT_DIR="$PROJECT_BASE/code/deepseek_agent/llm_backend/app/graphrag" \
nohup python run.py > "$PROJECT_BASE/runtime_data/backend.log" 2>&1 &

echo "等待后端启动..."
BACKEND_READY=0
for _ in {1..90}; do
  if curl --noproxy '*' --silent --fail --max-time 2 \
    http://127.0.0.1:8000/health > /dev/null; then
    BACKEND_READY=1
    break
  fi
  sleep 1
done

if [ "$BACKEND_READY" -eq 0 ]; then
  echo "警告: 后端在 90 秒内未就绪，请查看 $PROJECT_BASE/runtime_data/backend.log"
fi

echo ""
echo "=== 服务状态检查 ==="
python3 - <<'PY'
import socket
services = [
    (3306, 'MySQL'),
    (6379, 'Redis'),
    (7474, 'Neo4j HTTP'),
    (7687, 'Neo4j Bolt'),
    (8000, 'Backend API')
]
for port, name in services:
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect(('127.0.0.1', port))
        print(f'✓ {name:15} http://127.0.0.1:{port}')
    except:
        print(f'✗ {name:15} (端口 {port} 未响应)')
    finally:
        s.close()
PY

echo ""
echo "=== 访问地址 ==="
echo "前端页面: http://127.0.0.1:8000"
echo "API 文档: http://127.0.0.1:8000/docs"
echo "Neo4j 浏览器: http://127.0.0.1:7474 (用户名: neo4j, 密码见 llm_backend/.env)"
echo ""
echo "运行时数据目录: $PROJECT_BASE/runtime_data/"
echo "后端日志: $PROJECT_BASE/runtime_data/backend.log"
