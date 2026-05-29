#!/bin/bash
set -e

PORT=9500
IMAGE="scene-graph-nb"
CONTAINER="scene-graph-test"

echo "=== build ==="
docker build -t $IMAGE -f deploy/Dockerfile deploy/

echo "=== start ==="
docker rm -f $CONTAINER 2>/dev/null || true
docker run --rm -d -p $PORT:$PORT --name $CONTAINER $IMAGE
sleep 3

echo "=== health ==="
curl -s http://localhost:$PORT/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok', 'health fail'; print('ok'); print(json.dumps(d, indent=2))"

echo "=== model info ==="
curl -s http://localhost:$PORT/model_info | python3 -c "
import sys,json
d=json.load(sys.stdin)
assert d['object_classes']==150, 'wrong class count'
assert d['predicate_classes']==50, 'wrong pred count'
assert d['feature_counts']==[150,150,8,8,19,10,10], 'wrong feature counts'
print('ok')
print(f'  objects: {d[\"object_classes\"]} (sample: {d[\"sample_objects\"][:5]})')
print(f'  predicates: {d[\"predicate_classes\"]} (sample: {d[\"sample_predicates\"][:5]})')
print(f'  feature counts: {d[\"feature_counts\"]}')
print(f'  exist clf features: {d[\"exist_classifier_features\"]}')
print(f'  pred clf features: {d[\"pred_classifier_features\"]}')
"

echo "=== predict valid ==="
curl -s -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_width": 1024, "image_height": 1024,
    "pairs": [
      {"head_class":"person","tail_class":"horse","head_x1":100,"head_y1":50,"head_x2":300,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}
    ]
  }' | python3 -c "
import sys,json
d=json.load(sys.stdin)
assert len(d['predictions'])>0, 'no predictions'
p=d['predictions'][0]
assert p['head']=='person', f'wrong head: {p[\"head\"]}'
assert p['tail']=='horse', f'wrong tail: {p[\"tail\"]}'
print(f'ok: {p[\"head\"]} --{p[\"predicate\"]}--> {p[\"tail\"]} (conf={p[\"confidence\"]:.2f})')
print(json.dumps(d['predictions'][:3], indent=2))
"

echo "=== bad class ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_width": 1024, "image_height": 1024,
    "pairs": [{"head_class":"spaceship","tail_class":"horse","head_x1":100,"head_y1":50,"head_x2":300,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}]
  }')
BODY=$(curl -s -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_width": 1024, "image_height": 1024,
    "pairs": [{"head_class":"spaceship","tail_class":"horse","head_x1":100,"head_y1":50,"head_x2":300,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}]
  }')
[ "$STATUS" = "400" ] && echo "ok (400)" || echo "FAIL: expected 400 got $STATUS"
echo "$BODY" | python3 -m json.tool 2>/dev/null

echo "=== bad box ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_width": 1024, "image_height": 1024,
    "pairs": [{"head_class":"person","tail_class":"horse","head_x1":300,"head_y1":50,"head_x2":100,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}]
  }')
[ "$STATUS" = "400" ] && echo "ok (400)" || echo "FAIL: expected 400 got $STATUS"

echo "=== box out of bounds ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"image_width":100,"image_height":100,"pairs":[{"head_class":"person","tail_class":"horse","head_x1":0,"head_y1":0,"head_x2":300,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}]}')
[ "$STATUS" = "400" ] && echo "ok (400)" || echo "FAIL: expected 400 got $STATUS"

echo "=== negative coords ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"image_width":1024,"image_height":1024,"pairs":[{"head_class":"person","tail_class":"horse","head_x1":-10,"head_y1":50,"head_x2":300,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}]}')
[ "$STATUS" = "422" ] && echo "ok (422)" || echo "FAIL: expected 422 got $STATUS"

echo "=== empty pairs ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"image_width":1024,"image_height":1024,"pairs":[]}')
[ "$STATUS" = "422" ] && echo "ok (422)" || echo "FAIL: expected 422 got $STATUS"

echo "=== batch ==="
curl -s -X POST http://localhost:$PORT/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"image_width":1024,"image_height":1024,"pairs":[{"head_class":"person","tail_class":"horse","head_x1":100,"head_y1":50,"head_x2":300,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}]},
      {"image_width":1024,"image_height":1024,"pairs":[{"head_class":"cup","tail_class":"table","head_x1":350,"head_y1":250,"head_x2":400,"head_y2":320,"tail_x1":0,"tail_y1":300,"tail_x2":900,"tail_y2":600}]}
    ]
  }' | python3 -c "
import sys,json
d=json.load(sys.stdin)
assert len(d['results'])==2, 'wrong batch size'
print(f'ok: {len(d[\"results\"])} results')
for i,r in enumerate(d['results']):
    print(f'  image {i+1}:')
    for p in r['predictions'][:2]:
        print(f'    {p[\"head\"]} --{p[\"predicate\"]}--> {p[\"tail\"]} (conf={p[\"confidence\"]:.2f})')
"

echo "=== different dimensions ==="
curl -s -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_width": 500, "image_height": 375,
    "pairs": [
      {"head_class":"person","tail_class":"chair","head_x1":50,"head_y1":25,"head_x2":150,"head_y2":200,"tail_x1":25,"tail_y1":150,"tail_x2":150,"tail_y2":280}
    ]
  }' | python3 -c "
import sys,json
d=json.load(sys.stdin)
assert len(d['predictions'])>0, 'no predictions'
print(f'ok (500x375):')
for p in d['predictions'][:3]:
    print(f'  {p[\"head\"]} --{p[\"predicate\"]}--> {p[\"tail\"]} (conf={p[\"confidence\"]:.2f})')
"

echo "=== oversize dims ==="
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:$PORT/predict \
  -H "Content-Type: application/json" \
  -d '{"image_width":99999,"image_height":1024,"pairs":[{"head_class":"person","tail_class":"horse","head_x1":100,"head_y1":50,"head_x2":300,"head_y2":400,"tail_x1":250,"tail_y1":100,"tail_x2":800,"tail_y2":600}]}')
[ "$STATUS" = "422" ] && echo "ok (422)" || echo "FAIL: expected 422 got $STATUS"

echo "=== cleanup ==="
docker stop $CONTAINER 2>/dev/null || true

echo ""
echo "all passed"
