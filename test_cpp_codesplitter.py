import os
from typing import List

from llama_index.legacy.schema import Document, MetadataMode, TextNode
from llama_index.core.node_parser import CodeSplitter
def test_cpp_code_splitter() -> None:
    """Test case for code splitting using typescript."""
    if "CI" in os.environ:
        return

    code_splitter = CodeSplitter(
        language="cpp", chunk_lines=4, chunk_lines_overlap=1, max_chars=50
    )

    text = """\
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}"""

    chunks = code_splitter.split_text(text)
    print(chunks)
    assert chunks[0].startswith("#include <iostream>")
    assert chunks[1].startswith("int main()")
    assert chunks[2].startswith("{\n    std::cout")


test_cpp_code_splitter()