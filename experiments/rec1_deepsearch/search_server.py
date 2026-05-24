from flask import Flask, request, jsonify
from websearch_http import rollout_search
from webvisit_http import rollout_visit

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # 支持 JSON 格式的数据
        data = request.get_json()
        query = data.get('query') if data else None
    else:
        # 支持 GET 请求的 URL 参数
        query = request.args.get('query')

    if not query:
        return jsonify({'error': 'Missing "query" parameter'}), 400

    try:
        # 包装成 rollout_search 期望的字典格式
        # rollout_search 内部支持 string 或者 list 类型的 query
        params = {"query": query}
        response_text = rollout_search(params)
        
        return jsonify({
            'code': 0,
            'message': 'success',
            'data': response_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visit', methods=['GET', 'POST'])
def visit():
    if request.method == 'POST':
        data = request.get_json()
        url = data.get('url') if data else None
        goal = data.get('goal', 'Summarize the content of the page.') if data else 'Summarize the content of the page.'
    else:
        url = request.args.get('url')
        goal = request.args.get('goal', 'Summarize the content of the page.')

    if not url:
        return jsonify({'error': 'Missing "url" parameter'}), 400

    try:
        params = {"url": url, "goal": goal}
        response_text = rollout_visit(params)
        
        return jsonify({
            'code': 0,
            'message': 'success',
            'data': response_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 启动 Flask 服务，默认绑定 5000 端口
    app.run(host='0.0.0.0', port=5000, debug=True)


