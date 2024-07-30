# Learn GraphRAG

## Prepare

```
conda create -n learn-graphrag python=3.11 -y
conda activate learn-graphrag
```

```
pip install -r requirements.txt
```

## Neo4j

```
mkdir -p $HOME/neo4j/data
```

```
docker run \
    -d \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --name neo4j-apoc \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4J_PLUGINS=\[\"apoc\"\] \
    neo4j:5.21.2
```


## Run

```
python -m graphrag.index --root ./ragtest
```

```
python -m graphrag.query --root ./ragtest --method global "このストーリーの主題はなんですか"
python -m graphrag.query --root ./ragtest --method global "メイとサツキの関係を教えてください"
python -m graphrag.query --root ./ragtest --method global "大トトロはなんですか"
python -m graphrag.query --root ./ragtest --method local "メイとサツキの関係を教えてください"
python -m graphrag.query --root ./ragtest --method local "カンタはサツキに何を借りた？"
python -m graphrag.query --root ./ragtest --method local "サツキの妹はだれ？"
```