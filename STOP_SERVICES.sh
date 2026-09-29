#!/bin/bash
# 停止所有智能客服 Agent 服务

PROJECT_BASE=/home/dhc/projects/resume/Intelligent_Customer_Service_Agent

echo "=== 停止所有服务 ==="

echo "停止后端..."
pkill -f 'python run.py' || true

echo "停止 Neo4j..."
pkill -f "neo4j.*$PROJECT_BASE/runtime_data/neo4j_data" || true

echo "停止 Redis..."
pkill -f 'redis-server.*6379' || true

echo "停止 MySQL..."
pkill -f "mysqld.*$PROJECT_BASE/runtime_data/mysql_data" || true

sleep 3
echo "所有服务已停止"
