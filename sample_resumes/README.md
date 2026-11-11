# sample_resumes

All fixtures used by the tests are **synthetic**. Test PDFs are generated in code
(PyMuPDF `fitz.open()` + `page.insert_text(...)`) from invented names, `@example.com`
emails and fictitious phone numbers; nothing here comes from real people.

## Add a test PDF
```python
import pymupdf as fitz
doc = fitz.open()
page = doc.new_page()
for i, line in enumerate(["Jane Doe", "jane.doe@example.com", "SKILLS", "Python, SQL"]):
    page.insert_text((60, 60 + 16 * i), line)
doc.save("sample_resumes/synthetic_jane_doe.pdf")
```

## Rules
- Never commit real resumes or any personal data without the owner's authorization and consent.
- `.gitignore` excludes `sample_resumes/*.pdf`; keep personal PDFs local only.
- Use synthetic names, `example.com` emails and fictitious phone numbers.