
# COWORKING HANDBOOK QUESTION ANSWERING USING SECTION-ATTRIBUTED RAG

## PROBLEM STATEMENT
HarborWorks Coworking publishes its coworking membership rules as a single handbook. Members routinely need answers to narrow booking, access and service questions, and locating the governing clause by reading the handbook end to end is slow. An unattributed answer is of little practical use, because a decision on fees or access cannot be acted upon or appealed unless the member knows which section produced it.

The workspace requires a terminal application that answers a member's question from the handbook and reports the sections that support the answer.

---

## OBJECTIVE
Build a Retrieval-Augmented Generation pipeline that ingests the handbook PDF, divides it into overlapping chunks that retain their parent section, indexes those chunks in a ChromaDB vector store with section metadata, retrieves the excerpts most relevant to a member's question, and generates a grounded answer accompanied by the cited section titles.

---

## FILE STRUCTURE & PURPOSE OF FILES
* **`main.py`** — Implements the retrieval-augmented generation pipeline and the terminal execution block.
* **`tests.py`** — Contains the automated test cases that validate the implementation.
* **`requirements.txt`** — Lists the Python package dependencies required by the project.
* **`.env`** — Stores the Gemini API key and the model name as environment variables.
* **`data/coworking_handbook.pdf`** — Academic regulations handbook that serves as the knowledge base.

---

## DOCUMENT DESCRIPTION
The knowledge base is a three-page PDF titled *Coworking Member Handbook 2026-27 of HarborWorks Coworking*. It contains twelve numbered sections covering membership, desks, meeting rooms, visitors, internet, printing, kitchen facilities, lockers, fees, access, equipment, and conduct. Each section body contains 550 to 800 characters of fictional policy text for this assessment.

Every section heading occupies its own line and begins with a sequential number followed by the section title. Each heading is followed by a single paragraph of continuous prose stating the regulation. The document contains no tables, images, or multi-column layout, and every regulation carries specific figures such as percentages, fees, durations, and limits.

---

## TASK & FUNCTION SPECIFICATIONS

### 1. `load_and_chunk_document`

#### Function Signature
```python
load_and_chunk_document(document_path)

```

#### Parameters

* **`document_path`** (*String*): The relative path to the handbook PDF inside the `data` directory, supplied by the caller.

#### Purpose

Converts the handbook PDF into overlapping text chunks, each of which retains the title of the section it belongs to.

#### Requirements

* Extract the text of every page and treat it as one continuous body.
* Divide the body at its numbered section headings so that a heading and the paragraph beneath it form a single unit.
* Exclude any content appearing before the first heading, as it belongs to no section.
* Collapse line breaks inside each section body into single spaces before windowing.
* Reduce each section body to windows of 400 characters that advance in steps of 350 characters, so that consecutive windows of the same section share 50 characters.
* Confine every window to a single section.
* Discard any chunk shorter than 50 characters after stripping.

#### Returns

A list of dictionaries. Each dictionary holds the heading of the section the chunk belongs to and the chunk text, under the keys `section` and `text`.

---

### 2. `retrieve_sections`

#### Function Signature

```python
retrieve_sections(question, embedding_model, collection)

```

#### Parameters

* **`question`** (*String*): The natural language question typed by the member, received from the orchestrator.
* **`embedding_model`** (*SentenceTransformer instance*): The pre-loaded embedding model created in the orchestrator, used to convert the question into a vector of the same dimensionality as the stored chunk embeddings.
* **`collection`** (*ChromaDB collection object*): The in-memory collection created in the orchestrator, already populated with every chunk embedding, chunk text, and section label.

#### Purpose

Finds the handbook excerpts most semantically relevant to the question and returns them with their section labels and relevance scores.

#### Requirements

* Embed the question with the model supplied and query the collection for the three closest chunks.
* Read the returned documents, metadata, and distances together so that every chunk keeps its own section label.
* Convert each cosine distance into a similarity score (1 minus distance) rounded to two decimal places, where a higher value denotes a closer match.
* Return the results ordered most relevant first.

#### Returns

A list of three dictionaries ordered by descending similarity, each holding the excerpt text, its section title, and its similarity score, under the keys `chunk`, `section`, and `similarity`.

---

### 3. `augment_prompt`

#### Function Signature

```python
augment_prompt(question, retrieved_chunks)

```

#### Parameters

* **`question`** (*String*): The member's original question, passed through from the orchestrator.
* **`retrieved_chunks`** (*List of dictionaries*): The output of `retrieve_sections`, each entry carrying an excerpt, its section title, and its similarity score.

#### Purpose

Assembles the grounded prompt that will be submitted to the language model.

#### Requirements

* Present the excerpts as a numbered context block with each entry visibly labelled with the section it came from.
* Exclude the similarity score from the prompt.
* Include the original question in the prompt.
* Instruct the model to answer only from the supplied excerpts and to state the supporting section.
* Instruct the model to declare the matter uncovered when the excerpts do not contain the answer.

#### Returns

A string containing the complete prompt.

---

### 4. `generate_answer`

#### Function Signature

```python
generate_answer(prompt, client)

```

#### Parameters

* **`prompt`** (*String*): The assembled prompt produced by `augment_prompt`.
* **`client`** (*Gemini client object*): An authenticated client created in the orchestrator from the API key held in the environment.

#### Purpose

Submits the prompt to the language model and returns the generated answer.

#### Requirements

* Send the prompt through the client supplied.
* Read the model name from the environment rather than hard coding it.
* Return the response text free of leading and trailing whitespace.

#### Returns

A string containing the answer generated by the model.

---

### 5. `coworking_qa_pipeline`

#### Function Signature

```python
coworking_qa_pipeline(question, document_path)

```

#### Parameters

* **`question`** (*String*): The member's question, entered at the terminal or supplied directly by a caller.
* **`document_path`** (*String*): The relative path to the handbook PDF inside the `data` directory.

#### Purpose

Coordinates the complete pipeline and assembles the structured result, including the list of cited sections.

#### Requirements

* Chunk the handbook and embed every chunk in a single batch operation using the `all-MiniLM-L6-v2` `SentenceTransformer` model.
* Create an ephemeral in-memory ChromaDB collection with cosine distance, removing any collection surviving from an earlier call before creating it.
* Supply the embeddings explicitly rather than relying on the collection's own embedding function.
* Store each chunk with its text as the document and its section title as metadata, keeping identifiers, embeddings, documents, and metadata aligned.
* Run the three pipeline stages in order.
* Build the citation list from the section titles of the retrieved chunks with duplicates removed and retrieval order preserved.

#### Returns

A dictionary holding the original question, the retrieved chunks, the cited section titles, and the generated answer, under the keys `question`, `retrieved_chunks`, `sources`, and `answer`.

---

## MAIN BLOCK

### Requirements

* Read the question from the terminal and discard the surrounding whitespace.
* Reject an empty entry with a message and perform no further processing.
* Run the pipeline against the handbook.
* Print the question, the retrieved sections each with its similarity score, the comma-separated list of cited sections, and the generated answer.

---

## OUTPUT FORMAT

### Generated Files

No files are generated by this program. All results are printed to the console.

### Console Output

The program prints the question as entered, the three retrieved sections each with its similarity score, the comma-separated list of cited sections, and the generated answer. When the entry is empty, only the message `Question cannot be empty.` is printed.

### Expected Output Example

```text
Enter your question: How many meeting room hours are included in membership each month?

Question: How many meeting room hours are included in membership each month?

Retrieved Sections:
  1. 3. Meeting Room Booking (similarity: 0.67)
  2. 3. Meeting Room Booking (similarity: 0.48)
  3. 9. Membership Fees and Refunds (similarity: 0.31)

Sources: 3. Meeting Room Booking, 9. Membership Fees and Refunds

Answer:
Each active member receives 6 meeting room hours per calendar month. Unused hours expire
at month end and cannot be transferred. This is stated in section 3. Meeting Room Booking.

```

*Note: The similarity values and the wording of the generated answer vary between runs.*

---

## IMPLEMENTATION STEPS

### Environment Setup

1. **Navigate to project directory:**
```bash
cd Project

```


2. **Install dependencies from `requirements.txt`:**
```bash
pip install -r requirements.txt

```


3. **Implementing / Executing Solution:**
```bash
python3 main.py

```


4. **To Run Test Cases:**
```bash
python3 -m pytest tests.py -vv

```



## SECTION CONTENT

- **1. Membership and Onboarding**: Membership begins after identity verification, a signed agreement and payment of the first monthly invoice.
- **2. Desk Allocation and Reservations**: Hot-desk members may reserve one desk for up to 8 hours per day through the booking portal.
- **3. Meeting Room Booking**: Each active member receives 6 meeting room hours per calendar month.
- **4. Visitor Registration and Hosting**: Members may host up to 3 visitors at a time during staffed reception hours.
- **5. Internet and Network Use**: The shared network is available to active members through individual credentials issued during onboarding.
- **6. Printing and Scanning Services**: Members receive 100 black-and-white printed pages per calendar month.
- **7. Kitchen and Refreshment Facilities**: The shared kitchen is open from 8:00 a.m.
- **8. Lockers and Personal Storage**: Active members may rent one locker for INR 400 per calendar month, subject to availability.
- **9. Membership Fees and Refunds**: Monthly membership invoices are issued on the 1st and must be paid by the 5th calendar day of the month.
- **10. Access Hours and Building Security**: Members must leave the workspace by 10:00 p.m.
- **11. Equipment Borrowing and Late Fees**: A member may borrow up to 2 equipment items at a time for a period of 3 calendar days.
- **12. Conduct and Complaint Resolution**: Members must keep calls out of designated quiet zones and use headphones for audio at all times.

## PRACTICE NOTES

Implement the TODOs in `main.py`; no solution is supplied. Load `.env` using python-dotenv before model calls. Use `GEMINI_API_KEY` and `MODEL_NAME` for Gemini configuration. The 15 tests preserve the reference checks; they are not an exhaustive check of every prose requirement.
