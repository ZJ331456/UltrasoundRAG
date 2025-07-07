import re
from typing import List
from llama_index.core.schema import Document

class CustomMarkdownParser:
    """
    A custom parser for Markdown files that extracts image-text blocks.

    This parser identifies blocks where an image is associated with surrounding
    text paragraphs. It creates a single Document for each block, combining the
    text and storing the image URL in the metadata.
    """

    def __init__(self):
        """Initializes the parser."""
        # Regex to find markdown images with optional captions
        self.image_regex = re.compile(r"!\[(.*?)\]\((.*?)\)")

    def parse_file(self, file_path: str) -> List[Document]:
        """
        Parses a Markdown file and extracts image-text blocks.

        Args:
            file_path: The path to the Markdown file.

        Returns:
            A list of Document objects, where each document represents an
            image-text block or a standalone text block.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        # Split the document into paragraphs or blocks
        blocks = content.split('\n\n')
        
        i = 0
        while i < len(blocks):
            block = blocks[i].strip()
            if not block:
                i += 1
                continue

            match = self.image_regex.search(block)
            
            if match:
                # This block contains an image.
                image_caption = match.group(1).strip()
                image_url = match.group(2).strip()
                
                # Context is the paragraph before the image.
                # And the paragraph after the image (often a figure description).
                context_before = blocks[i-1].strip() if i > 0 and not blocks[i-1].strip().startswith('#') else ''
                context_after = blocks[i+1].strip() if (i + 1) < len(blocks) and not blocks[i+1].strip().startswith('#') else ''
                
                # The image line itself might contain text, let's remove the markdown for the image
                image_line_text = self.image_regex.sub('', block).strip()

                combined_text = f"{context_before}\n{image_line_text}\n{image_caption}\n{context_after}".strip()
                
                doc = Document(
                    text=combined_text,
                    metadata={
                        "source": file_path,
                        "image_url": image_url,
                        "image_caption": image_caption,
                        "block_type": "image_text"
                    }
                )
                documents.append(doc)
                
                # Skip the next block if it was used as 'context_after'
                if context_after:
                    i += 2
                else:
                    i += 1
            else:
                # This is a regular text block
                doc = Document(
                    text=block,
                    metadata={
                        "source": file_path,
                        "block_type": "text"
                    }
                )
                documents.append(doc)
                i += 1

        return documents

def main():
    """Main function for testing the parser."""
    parser = CustomMarkdownParser()
    
    # Create a dummy markdown file for testing
    dummy_md_content = """# Section 1

This is the first introductory paragraph. It has some important general information.

This is the descriptive text for the first image. It explains what the ultrasound shows.

![Doppler ultrasound of the carotid artery](images/01_超声标准切面图解/image-20240618153455325.png)

Figure 1: This shows the standard view of the carotid artery.

## Subsection

This paragraph is under a subsection and not directly related to an image.

Here is another image, but without preceding text.

![Second image with no context before](images/01_超声标准切面图解/image-20240618153500000.png)

This is the caption for the second image.

And a final text block.
"""
    dummy_file_path = "dummy_test_file.md"
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write(dummy_md_content)
        
    print(f"--- Parsing {dummy_file_path} ---")
    documents = parser.parse_file(dummy_file_path)
    
    for doc in documents:
        print("--- New Document ---")
        print(f"Text: {doc.text}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 20)
        
    # Clean up the dummy file
    import os
    os.remove(dummy_file_path)

if __name__ == "__main__":
    main() 