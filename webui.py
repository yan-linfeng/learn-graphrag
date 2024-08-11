import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import oracledb
import yaml
from dotenv import load_dotenv, find_dotenv
from oracledb import DatabaseError
from unstructured.partition.auto import partition

from graphrag.config import create_graphrag_config
from graphrag.index import create_pipeline_config, run_pipeline_with_config
from graphrag.index.emit import TableEmitterType
from graphrag.index.progress.rich import RichProgressReporter
from utils.generator_util import generate_unique_id

custom_css = """
body {
  font-family: "Noto Sans JP", Arial, sans-serif !important;
}

/* Hide sort buttons at gr.DataFrame */
.sort-button {
    display: none !important;
} 

body gradio-app .tabitem .block{
    background: #fff !important;
}

.gradio-container{
    background: #c4c4c440;
}

.tabitem .form{
border-radius: 3px;
}

.main_Header>span>h1{
    color: #fff;
    text-align: center;
    margin: 0 auto;
    display: block;
}

.tab-nav{
    # border-bottom: none !important;
}

.tab-nav button[role="tab"]{
    color: rgb(96, 96, 96);
    font-weight: 500;
    background: rgb(255, 255, 255);
    padding: 10px 20px;
    border-radius: 4px 4px 0px 0px;
    border: none;
    border-right: 4px solid gray;
    border-radius: 0px;
    min-width: 150px;
}

.tabs .tabitem .tabs .tab-nav button[role="tab"]{
    min-width: 90px;
    padding: 5px;
    border-right: 1px solid #186fb4;
    border-top: 1px solid #186fb4;
    border-bottom: 0.2px solid #fff;
    margin-bottom: -2px;
    z-index: 3;
}


.tabs .tabitem .tabs .tab-nav button[role="tab"]:first-child{
    border-left: 1px solid #186fb4;
        border-top-left-radius: 3px;
}

.tabs .tabitem .tabs .tab-nav button[role="tab"]:last-child{
    border-right: 1px solid #186fb4;
}

.tab-nav button[role="tab"]:first-child{
       border-top-left-radius: 3px;
}

.tab-nav button[role="tab"]:last-child{
        border-top-right-radius: 3px;
    border-right: none;
}
.tabitem{
    background: #fff;
    border-radius: 0px 3px 3px 3px !important;
    box-shadow: rgba(50, 50, 93, 0.25) 0px 2px 5px -1px, rgba(0, 0, 0, 0.3) 0px 1px 3px -1px;
}

.tabitem .tabitem{
    border: 1px solid #196fb4;
    background: #fff;
    border-radius: 0px 3px 3px 3px !important;
}

.tabitem textarea, div.tabitem div.container>.wrap{
    background: #f4f8ffc4;
}

.tabitem .container .wrap {
    border-radius: 3px;
}

.tab-nav button[role="tab"].selected{
    color: #fff;
    background: #196fb4;
    border-bottom: none;
}

.tabitem .inner_tab button[role="tab"]{
   border: 1px solid rgb(25, 111, 180);
   border-bottom: none;
}

.app.gradio-container {
  max-width: 1440px;
}

gradio-app{
    background-image: url("https://objectstorage.ap-tokyo-1.oraclecloud.com/n/sehubjapacprod/b/km_newsletter/o/tmp%2Fmain_bg.png") !important;
    background-size: 100vw 100vh !important;
}

input, textarea{
    border-radius: 3px;
}


.container>input:focus, .container>textarea:focus, .block .wrap .wrap-inner:focus{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

.tabitem div>button.primary{
    border: none;
    background: linear-gradient(to bottom right, #ffc679, #f38141);
    color: #fff;
    box-shadow: 2px 2px 2px #0000001f;
    border-radius: 3px;
}

.tabitem div>button.primary:hover{
    border: none;
    background: #f38141;
    color: #fff;
    border-radius: 3px;
    box-shadow: 2px 2px 2px #0000001f;
}


.tabitem div>button.secondary{
    border: none;
    background: linear-gradient(to right bottom, rgb(215 215 217), rgb(194 197 201));
    color: rgb(107 106 106);
    box-shadow: rgba(0, 0, 0, 0.12) 2px 2px 2px;
    border-radius: 3px;
}

.tabitem div>button.secondary:hover{
    border: none;
    background: rgb(175 175 175);
    color: rgb(255 255 255);
    border-radius: 3px;
    box-shadow: rgba(0, 0, 0, 0.12) 2px 2px 2px;
}

.cus_ele1_select .container .wrap:focus-within{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

input[type="checkbox"]:checked, input[type="checkbox"]:checked:hover, input[type="checkbox"]:checked:focus {
    border-color: #186fb4;
    background-color: #186fb4;
}

#event_tbl{
    border-radius:3px;
}

#event_tbl .table-wrap{
    border-radius:3px;
}

#event_tbl table thead>tr>th{
    background: #bfd1e0;
        min-width: 90px;
}

#event_tbl table thead>tr>th:first-child{
    border-radius:3px 0px 0px 0px;
}
#event_tbl table thead>tr>th:last-child{
    border-radius:0px 3px 0px 0px;
}


#event_tbl table .cell-wrap span{
    font-size: 0.8rem;
}

#event_tbl table{
    overflow-y: auto;
    overflow-x: auto;
}

#event_exp_tbl .table-wrap{
     border-radius:3px;   
}
#event_exp_tbl table thead>tr>th{
    background: #bfd1e0;
}

.count_t1_text .prose{
    padding: 5px 0px 0px 6px;
}

.count_t1_text .prose>span{
    padding: 0px;
}

.cus_ele1_select .container .wrap:focus-within{
    border-radius: 3px;
    box-shadow: rgb(255 246 228 / 63%) 0px 0px 0px 3px, rgb(255 248 236 / 12%) 0px 2px 4px 0px inset !important;
    border-color: rgb(249 169 125 / 87%) !important;
}

.count_t1_text .prose>span{
    font-size: 0.9rem;
}


footer{
  display: none !important;
}

.sub_Header>span>h3,.sub_Header>span>h2,.sub_Header>span>h4{
    color: #fff;
    font-size: 0.8rem;
    font-weight: normal;
    text-align: center;
    margin: 0 auto;
    padding: 5px;
}

@media (min-width: 1280px) {
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1400px;
    }
}
.gap.svelte-vt1mxs{
    gap: unset;
}

.tabitem .gap.svelte-vt1mxs{
        gap: var(--layout-gap);
}

@media (min-width: 1280px) {
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1400px;
    }
}

"""

# read local .env file
load_dotenv(find_dotenv())

DEFAULT_COLLECTION_NAME = os.environ["DEFAULT_COLLECTION_NAME"]

# if platform.system() == 'Linux':
#     oracledb.init_oracle_client(lib_dir="/u01/aipoc/instantclient_23_5")

# 初始化一个数据库连接
pool = oracledb.create_pool(
    dsn=os.environ["ORACLE_AI_CONNECTION_STRING"],
    min=1,
    max=5,
    increment=1
)


def delete_all_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def get_doc_list() -> List[Tuple[str, str]]:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            try:
                select_sql = f"""
SELECT
    json_value(cmetadata, '$.file_name') name,
    id
FROM
    {DEFAULT_COLLECTION_NAME}_collection
ORDER BY name 
"""
                cursor.execute(select_sql)
                return [(f"{row[0]}", row[1]) for row in cursor.fetchall()]
            except DatabaseError as de:
                return []


def refresh_doc_list():
    doc_list = get_doc_list()
    return (
        gr.Radio(choices=doc_list, value=""),
        gr.CheckboxGroup(choices=doc_list, value="")
    )


def get_server_path(doc_id: str) -> str:
    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            select_sql = f"""
SELECT json_value(cmetadata, '$.server_path') AS server_path
FROM {DEFAULT_COLLECTION_NAME}_collection
WHERE id = :doc_id
"""
            cursor.execute(select_sql, doc_id=doc_id)
            return cursor.fetchone()[0]


def redact(input: dict) -> str:
    """Sanitize the config json."""

    # Redact any sensitive configuration
    def redact_dict(input: dict) -> dict:
        if not isinstance(input, dict):
            return input

        result = {}
        for key, value in input.items():
            if key in {
                "api_key",
                "connection_string",
                "container_name",
                "organization",
            }:
                if value is not None:
                    result[key] = f"REDACTED, length {len(value)}"
            elif isinstance(value, dict):
                result[key] = redact_dict(value)
            elif isinstance(value, list):
                result[key] = [redact_dict(i) for i in value]
            else:
                result[key] = value
        return result

    redacted_dict = redact_dict(input)
    return json.dumps(redacted_dict, indent=4)


def load_graph_document(file_path, server_directory):
    if not file_path:
        raise gr.Error("ファイルを選択してください")
    if not os.path.exists(server_directory):
        os.makedirs(server_directory)
    doc_id = generate_unique_id("doc_")
    file_name = os.path.basename(file_path.name)
    server_path = os.path.join(server_directory, f"{doc_id}_{file_name}")
    shutil.copy(file_path.name, server_path)
    elements = partition(filename=server_path, strategy='fast',
                         languages=["jpn", "eng", "chi_sim"],
                         extract_image_block_types=["Table"],
                         extract_image_block_to_payload=False,
                         # skip_infer_table_types=["pdf", "ppt", "pptx", "doc", "docx", "xls", "xlsx"])
                         skip_infer_table_types=["pdf", "jpg", "png", "heic", "doc", "docx"])
    original_contents = "\n\n".join([str(el) for el in elements])
    print(f"{original_contents=}")

    collection_cmeta = {
        'file_name': file_name,
        'source': server_path,
        'server_path': server_path,
    }

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.setinputsizes(**{"data": oracledb.DB_TYPE_BLOB})
            load_document_sql = f"""
    -- (Only for Reference) Insert to table {DEFAULT_COLLECTION_NAME}_collection 
    INSERT INTO {DEFAULT_COLLECTION_NAME}_collection(id, data, cmetadata)
    VALUES (:id, to_blob(:data), :cmetadata) 
    """
            output_sql_text = load_document_sql.replace(":id", "'" + str(doc_id) + "'")
            output_sql_text = output_sql_text.replace(":data", "'...'")
            output_sql_text = output_sql_text.replace(":cmetadata", "'" + json.dumps(collection_cmeta) + "'") + ";"
            print(f"{output_sql_text=}")
            cursor.execute(load_document_sql, {
                'id': doc_id,
                'data': original_contents,
                'cmetadata': json.dumps(collection_cmeta)
            })
            conn.commit()

    return gr.Textbox(value=doc_id), gr.Textbox(value=original_contents)


async def create_index(doc_id):
    if not doc_id:
        raise gr.Error("ドキュメントを選択してください")

    root = "./data"
    delete_all_files_in_directory(root + "/input")
    doc_server_path = get_server_path(doc_id)
    shutil.copy(doc_server_path, root + "/input/")

    _root = Path(root)
    config = "./settings.yaml"
    settings_yaml = (
        Path(config)
        if config and Path(config).suffix in [".yaml", ".yml"]
        else _root / "settings.yaml"
    )
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"
    if settings_yaml.exists():
        progress_reporter = RichProgressReporter("GraphRAG Indexer ")
        progress_reporter.success(f"Reading settings from {settings_yaml}")
        with settings_yaml.open("rb") as file:
            data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
            parameters = create_graphrag_config(data, root)
            progress_reporter.info(f"Using default configuration: {redact(parameters.model_dump())}")
            pipeline_config = create_pipeline_config(parameters)
            progress_reporter.info(f"Final Config: {redact(pipeline_config.model_dump())}")

            run_id = time.strftime("%Y%m%d-%H%M%S")
            # emit = 'parquet,csv'
            emit = 'parquet'
            pipeline_emit = emit.split(',') if emit else None
            response = ""
            book_paper_emoji = ["📗", "📙"]
            book_paper_i = 1
            async for output in run_pipeline_with_config(
                pipeline_config,
                run_id=run_id,
                memory_profile=False,
                cache=None,
                progress_reporter=progress_reporter,
                emit=(
                    [TableEmitterType(e) for e in pipeline_emit]
                    if pipeline_emit
                    else None
                ),
                is_resume_run=False,
            ):
                if output.errors and len(output.errors) > 0:
                    progress_reporter.error(output.workflow)
                    raise ValueError(f"Encountered errors: {output.errors}")
                else:
                    progress_reporter.success(output.workflow)
                    response += str(f"{book_paper_emoji[book_paper_i % 2]} {output.workflow}") + "\n\n"
                    book_paper_i += 1

                progress_reporter.info(str(output.result))
                response += str(output.result) + "\n\n"
                yield response
            progress_reporter.stop()


def merge_index(doc_id):
    if not doc_id:
        raise gr.Error("ドキュメントを選択してください")

    with pool.acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT chunk_text, chunk_vector, attributes FROM graph_rag")
            rows = cursor.fetchall()

            insert_sql = """
            INSERT INTO langchain_oracle_embedding (doc_id, embed_id, embed_data, embed_vector, cmetadata)
            VALUES (:1, :2, :3, :4, :5)
            """

            for i, row in enumerate(rows, start=1):
                chunk_text, chunk_vector, attributes = row
                cursor.execute(insert_sql, (doc_id, i, chunk_text, chunk_vector, attributes))

            cursor.execute("TRUNCATE TABLE graph_rag")

            conn.commit()

    return gr.Textbox()


def delete_document(server_directory, doc_ids):
    if not server_directory:
        raise gr.Error("サーバー・ディレクトリを入力してください")
    print(f"{doc_ids=}")
    if not doc_ids or len(doc_ids) == 0 or (len(doc_ids) == 1 and doc_ids[0] == ''):
        raise gr.Error("ドキュメントを選択してください")

    output_sql = ""
    with pool.acquire() as conn, conn.cursor() as cursor:
        for doc_id in filter(bool, doc_ids):
            server_path = get_server_path(doc_id)
            if os.path.exists(server_path):
                os.remove(server_path)
                print(f"File {doc_id} deleted successfully")
            else:
                print(f"File {doc_id} not found")

            delete_embedding_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_embedding
WHERE doc_id = :doc_id 
"""
            delete_collection_sql = f"""
DELETE FROM {DEFAULT_COLLECTION_NAME}_collection
WHERE id = :doc_id 
"""
            output_sql += delete_embedding_sql.strip().replace(":doc_id", "'" + doc_id + "'") + "\n"
            output_sql += delete_collection_sql.strip().replace(":doc_id", "'" + doc_id + "'")
            print(f"{output_sql=}")
            cursor.execute(delete_embedding_sql, doc_id=doc_id)
            cursor.execute(delete_collection_sql, doc_id=doc_id)

            conn.commit()

    doc_list = get_doc_list()
    return gr.CheckboxGroup(choices=doc_list, value="")


with gr.Blocks(css=custom_css) as app:
    # webs
    gr.Markdown(value="# RAG精度あげたろう", elem_classes="main_Header")
    gr.Markdown(value="### LLM&RAG精度検証ツール", elem_classes="sub_Header")
    with gr.Tabs() as tabs:
        with gr.TabItem(label="環境設定") as tab_setting:
            with gr.TabItem(label="OCI GenAIの設定*") as tab_create_oci_cred:
                pass
        with gr.TabItem(label="GraphRAG評価", elem_classes="inner_tab") as tab_graphrag_evaluation:
            with gr.TabItem(label="Step-1.読込み") as tab_graphrag_load_document:
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_load_document_doc_id_text = gr.Textbox(label="Doc ID", lines=1,
                                                                            interactive=False)
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_load_document_page_content_text = gr.Textbox(label="コンテンツ", lines=15,
                                                                                  max_lines=15,
                                                                                  autoscroll=False,
                                                                                  show_copy_button=True,
                                                                                  interactive=False)
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_load_document_file_text = gr.File(label="ファイル*",
                                                                       file_types=["txt"],
                                                                       type="filepath")
                    with gr.Column():
                        tab_graphrag_load_document_server_directory_text = gr.Text(label="サーバー・ディレクトリ*",
                                                                                   value="/u01/data/graphrag/")
                with gr.Row(visible=True):
                    with gr.Column():
                        gr.Examples(examples=[os.path.join(os.path.dirname(__file__), "files/xiyouji.txt")],
                                    label="サンプル・ファイル",
                                    inputs=tab_graphrag_load_document_file_text)
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_load_document_load_button = gr.Button(value="読込み", variant="primary")

            with gr.TabItem(label="Step-2.Graphの作成") as tab_graphrag_create_index:
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_create_index_status_text = gr.Textbox(label="ステータス", lines=20,
                                                                           interactive=False, autoscroll=True)
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_create_index_doc_id_radio = gr.Radio(
                            choices=get_doc_list(),
                            label="ドキュメント*"
                        )
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_create_index_button = gr.Button(
                            value="Graphの作成（時間がかかる）",
                            variant="primary")

            with gr.TabItem(label="Step-3.削除(オプション)") as tab_graphrag_delete_document:
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_delete_document_server_directory_text = gr.Text(
                            label="サーバー・ディレクトリ*",
                            value="/u01/data/graphrag/")
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_delete_document_doc_ids_checkbox_group = gr.CheckboxGroup(
                            choices=get_doc_list(),
                            label="ドキュメント*"
                        )
                with gr.Row():
                    with gr.Column():
                        tab_graphrag_delete_document_delete_button = gr.Button(value="削除", variant="primary")

    gr.Markdown(value="### Developed by Oracle Japan", elem_classes="sub_Header")

    # actions
    tab_graphrag_load_document_load_button.click(load_graph_document,
                                                 inputs=[tab_graphrag_load_document_file_text,
                                                         tab_graphrag_load_document_server_directory_text],
                                                 outputs=[tab_graphrag_load_document_doc_id_text,
                                                          tab_graphrag_load_document_page_content_text],
                                                 )
    tab_graphrag_create_index.select(refresh_doc_list,
                                     outputs=[tab_graphrag_create_index_doc_id_radio,
                                              tab_graphrag_delete_document_doc_ids_checkbox_group])
    tab_graphrag_create_index_button.click(create_index,
                                           inputs=[tab_graphrag_create_index_doc_id_radio],
                                           outputs=[tab_graphrag_create_index_status_text]
                                           ).then(merge_index, inputs=[tab_graphrag_create_index_doc_id_radio],
                                                  outputs=[tab_graphrag_create_index_status_text])
    tab_graphrag_delete_document.select(refresh_doc_list,
                                        outputs=[tab_graphrag_create_index_doc_id_radio,
                                                 tab_graphrag_delete_document_doc_ids_checkbox_group])
    tab_graphrag_delete_document_delete_button.click(delete_document,
                                                     inputs=[tab_graphrag_delete_document_server_directory_text,
                                                             tab_graphrag_delete_document_doc_ids_checkbox_group],
                                                     outputs=[tab_graphrag_delete_document_doc_ids_checkbox_group])

app.queue()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        max_threads=200,
        show_api=False,
        # auth=do_auth,
    )
