from pymilvus import connections

try:
    connections.connect(host='localhost', port='19530')
    print('Successfully connected to Milvus')
    connections.disconnect()
except Exception as e:
    print(f'Failed to connect: {str(e)}')