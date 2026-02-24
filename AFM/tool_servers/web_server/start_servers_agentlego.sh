#!/bin/bash

# =====================================================================================================================
#                                      Environment setup
# =====================================================================================================================
: "${SERVER_HOST:?Please set SERVER_HOST in environment.sh}"
: "${CRAWL_PAGE_PORT:?Please set CRAWL_PAGE_PORT in environment.sh}"
: "${WEBSEARCH_PORT:?Please set WEBSEARCH_PORT in environment.sh}"
: "${AGENTLEGO_TOOL_PORT:?Please set AGENTLEGO_TOOL_PORT in environment.sh}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DIR/logs/$SERVER_HOST";   mkdir -p "$LOG_DIR"

cmd=$1
if [[ ! "$cmd" =~ ^(start|stop|status|test)$ ]]; then
  echo "Usage: $0 [start|stop|status|test]"
  echo "  start  : Start servers (SerperV2, CrawlPageV2, AgentLegoToolServerV2)"
  echo "  stop   : Stop all servers"
  echo "  status : Check server status"
  echo "  test   : Test server functionality"
  exit 1
fi

# =====================================================================================================================
#                                      start
# =====================================================================================================================
if [[ "$cmd" == "start" ]]; then
  echo "Starting servers..."

  # CrawlPageV2
  logf="$LOG_DIR/CrawlPageV2_$CRAWL_PAGE_PORT.log"
  if netstat -tulnp | grep -q ":$CRAWL_PAGE_PORT "; then
    echo "CrawlPageV2 already running on port $CRAWL_PAGE_PORT"
  else
    echo "Starting CrawlPageV2 on port $CRAWL_PAGE_PORT..."
    nohup python -u "$DIR/v2/crawl_page_server_v2.py" > "$logf" 2>&1 &
  fi

  # SerperCacheV2
  logf="$LOG_DIR/SerperCacheV2_$WEBSEARCH_PORT.log"
  if netstat -tulnp | grep -q ":$WEBSEARCH_PORT "; then
    echo "SerperCacheV2 already running on port $WEBSEARCH_PORT"
  else
    echo "Starting SerperCacheV2 on port $WEBSEARCH_PORT..."
    nohup python -u "$DIR/v2/cache_serper_server_v2.py" > "$logf" 2>&1 &
  fi

  # AgentLegoToolServerV2
  logf="$LOG_DIR/AgentLegoToolServer_$AGENTLEGO_TOOL_PORT.log"
  if netstat -tulnp | grep -q ":$AGENTLEGO_TOOL_PORT "; then
    echo "AgentLegoToolServerV2 already running on port $AGENTLEGO_TOOL_PORT"
  else
    echo "Starting AgentLegoToolServerV2 on port $AGENTLEGO_TOOL_PORT..."
    CUDA_VISIBLE_DEVICES=2 nohup python -u "$DIR/v2/agentlego_server_v2.py" > "$logf" 2>&1 &
  fi

# =====================================================================================================================
#                                      test
# =====================================================================================================================
elif [[ "$cmd" == "test" ]]; then
  echo "-------------------- Testing SerperCacheV2 --------------------"
  python -u "$DIR/server_tests/test_cache_serper_server_v2.py" "http://$SERVER_HOST:$WEBSEARCH_PORT/search"
  echo "-------------------- Testing CrawlPageV2 ----------------------"
  python -u "$DIR/server_tests/test_crawl_page_simple_v2.py" "http://$SERVER_HOST:$CRAWL_PAGE_PORT/crawl_page"
  echo "-------------------- Testing AgentLegoToolServerV2 ------------"
  curl -X POST "http://$SERVER_HOST:$AGENTLEGO_TOOL_PORT/run" \
    -H "Content-Type: application/json" \
    -d '{"tool":"image_description","query":"{\"image\":\"/share/softwares/haider/Agent_Foundation_Models/AFM/data/web_agent/GTA/image/image_1.jpg\"}"}'
  echo
  echo "-------------------- All Tests Completed -----------------------"

# =====================================================================================================================
#                                      stop
# =====================================================================================================================
elif [[ "$cmd" == "stop" ]]; then
  echo "Stopping all servers..."

  stop_by_port() {
    local port=$1
    local name=$2
    local port_info
    port_info=$(netstat -tulnp | grep ":$port ")
    if [[ -n "$port_info" ]]; then
      pids=($(echo "$port_info" | awk '{print $7}' | grep -o '^[0-9]*' | sort -u))
      echo "Found ${#pids[@]} process(es) for $name on port $port"
      for pid in "${pids[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
          echo "Stopping $name (PID $pid)"
          kill "$pid" 2>/dev/null
          for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
              echo "Stopped $name (PID $pid)"
              break
            fi
            sleep 0.5
          done
          if kill -0 "$pid" 2>/dev/null; then
            echo "Force stopping $name (PID $pid)"
            kill -9 "$pid" 2>/dev/null
          fi
        fi
      done
    else
      echo "No process found for $name on port $port"
    fi
  }

  stop_by_port "$CRAWL_PAGE_PORT" "CrawlPageV2"
  stop_by_port "$WEBSEARCH_PORT" "SerperCacheV2"
  stop_by_port "$AGENTLEGO_TOOL_PORT" "AgentLegoToolServerV2"

# =====================================================================================================================
#                                      status
# =====================================================================================================================
else
  echo "Server status:"
  check_port() {
    local port=$1
    local name=$2
    if netstat -tulnp | grep -q ":$port "; then
      echo "$name is running on port $port"
    else
      echo "$name is not running"
    fi
  }

  check_port "$CRAWL_PAGE_PORT" "CrawlPageV2"
  check_port "$WEBSEARCH_PORT" "SerperCacheV2"
  check_port "$AGENTLEGO_TOOL_PORT" "AgentLegoToolServerV2"
fi
