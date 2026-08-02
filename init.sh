protoc --proto_path=. --python_out=tf proto/net.proto; protoc --proto_path=. --python_out=tf proto/chunk.proto; touch tf/proto/__init__.py
