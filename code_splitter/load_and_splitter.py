from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
def load_and_split(repo_path):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".cpp", ".h"],
        parser=LanguageParser(language=Language.CPP),
    )
    data = loader.load()
    print(len(data))
    contents = [document.page_content for document in data]

    # 用分隔符 '\n\n--8<--\n\n' 连接所有文档内容
    combined_contents = "\n\n--8<--\n\n".join(contents)

    # 将连接后的内容写入文件
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(combined_contents)
    cpp_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=3500, chunk_overlap=200)
    all_splits = cpp_splitter.split_documents(data)
    print(len(all_splits))
    return all_splits
