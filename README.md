# Docs4AI

## Overview

Docs4AI provides condensed, optimized documentation specifically tailored for efficient consumption by Large Language Models (LLMs). Traditional documentation, primarily designed for humans, often includes redundant content, extraneous formatting, and promotional material that unnecessarily occupies valuable context window space in LLMs. This project aims to create streamlined documentation, maximizing the efficiency and effectiveness of LLM-assisted programming tasks.

## Origin

This project originated from the following tweet:

[https://x.com/doodlestein/status/1900552731762253908](https://x.com/doodlestein/status/1900552731762253908)

> Just had an interesting idea for a useful service: go through all the python packages available from pip, in order of popularity.
>
> For each one, take the full, latest documentation website and feed it into Claude 3.7, section by section, asking it to put it into a much denser, plain-text form best suited for consumption by other LLMs for coding tasks.
>
> Then make a very simple website that serves these text files, like:
>
> docs4ai dot com slash pandas
>
> Then use function calling or MCP to make it easy for any model to grab these docs on demand as needed. The main insight here is that human docs have a ton of fluff that isn't really necessary and wastes context window space.
>
> And often it's also presented with a lot of formatting that makes it harder for the LLM to process, causing it to waste its cognition on silly, low-value stuff like fixing html tables that got mangled during copy/paste operations.
>
> I guess you'd want to do it for all languages, so you'd have:
>
> docs4ai dot com slash python slash pandas
> docs4ai dot com slash js slash tailwindcss
> docs4ai dot com slash rust slash serde
>
> Not sure you could charge for that. And the LLMs don't have money to spend (yet), so you can't show them ads (not that you could really show ads serving plain text files, anyway). But it would certainly be useful to a lot of people!

## Methodology

The documentation optimization process involves:

1. **Collection**: Gathering comprehensive documentation for popular open-source libraries.

2. **Distillation**: Employing advanced LLMs, such as Claude 3.7, to meticulously analyze, interpret, and rewrite the documentation into a more compact, clear, and LLM-friendly format.

3. **Organization**: Structuring the documentation consistently and logically across various programming languages and libraries.

### Distillation Focus Areas

- Eliminating redundant explanations and marketing content
- Retaining all critical technical details
- Prioritizing API references, usage examples, and common errors
- Minimal markdown or plaintext formatting to facilitate easy parsing
- Ensuring completeness while conserving token usage

## Current Implementation: Marqo Example

This repository initially showcases optimized documentation for the `marqo` Python library, which enables tensor-based document indexing and combined lexical/semantic search.

### Files Included

- [`marqo_original_docs_merged.txt`](https://github.com/DimitriGilbert/docs4ai/blob/main/marqo_original_docs_merged.txt): Original, full documentation from Marqo.
- [`marqo_condensed_docs_for_lllm.md`](https://github.com/DimitriGilbert/docs4ai/blob/main/marqo_condensed_docs_for_lllm.md): Condensed, optimized version specifically crafted for LLM consumption.

### Condensation Strategy

The distillation process used the following prompting strategy with Claude 3.7:

- Provided complete Marqo documentation with instructions to exclude cloud-specific elements.
- Directed Claude to meticulously read, understand, and then distill the content to its most compact yet clear form for model comprehension.
- Managed the extensive documentation by dividing it into five sections, followed by an additional sixth section to capture any overlooked details.
- Permitted content reorganization to optimize for LLM efficiency over human readability.

### Actual Prompts Used for Marqo Example:

*Main Prompt:*

> I attached the COMPLETE documentation for marqo, the python library and system that allows you to index documents and do fused lexical/semantic search over them.
>
> The documentation contains a lot of "fluff" and references to the Marqo cloud service (api keys, etc.) which we do NOT care about-- we only care about the open source marqo software that you run locally on Linux yourself. I pasted the docs manually from the documentation website.
>
> Your goal is to meticulously read and understand everything and then distill it down into the most compact and clear form that would still be easily intelligible to YOU in a new conversation where you had never seen the original documentation text. It's too long to do in one shot, so let's do it in 5 sections, starting now with the first part.
>
> Feel free to rearrange the order of sections so that it makes most sense to YOU since YOU will be the one consuming the output! Don't worry if it would be confusing or hard to follow by a human, it's not for a human, it's for YOU to quickly and efficiently convey all the relevant information in the documentation.

*Follow-Up Prompts of the Form:*

> continue with part 2 of 5
>
> ...
>
> continue with part 4 of 5
>
> ok now make a part 6 with all the important stuff that you left out from the full docs in the 5 parts you already wrote above

## Benefits and Use Cases

Optimized documentation offers multiple advantages:

- **Efficient Context Utilization**: Reduces unnecessary information, maximizing relevant content within an LLM's context window.
- **Enhanced Reasoning**: Minimizes cognitive overhead, allowing the model to focus purely on problem-solving tasks.
- **Reduced Hallucinations**: Provides accurate, concise references to prevent incorrect inferences.
- **Improved Tool Integration**: Ideal for integration with LLM-based coding assistants or automated retrieval systems.

Common use cases include:

- AI-driven programming tools and assistants
- On-demand documentation retrieval via API calls or function-calling frameworks
- Enhanced educational resources for AI-guided coding tutorials

Below is an expanded “App” section that details the project’s architecture, explains step‑by‑step what needs to be implemented, and proposes a comprehensive data model and a suite of type‑safe TRPC endpoints. In addition, we’ve added extra functionality ideas that are both relevant and practical.

---

## App

This section describes the final configuration for the Next.js app found in the **site/** folder and explains the overall workflow, data structures, API endpoints, and extra features to enhance the Docs4AI experience.

### 1. Project Setup and Architecture

- **Stack Overview**:
  - **Frontend**: Next.js with Tailwind CSS and shadcn components for a modern, responsive UI.
  - **Backend**:
    - **TRPC** to provide type‑safe API endpoints.
    - **Drizzle ORM** for interacting with the database.
    - **NextAuth** for authentication (supporting OAuth and “bring your own API key” flows).
  - **LLM Integration**: Uses [Vercel AI-SDK](https://sdk.vercel.ai/docs/) to communicate with LLM providers (e.g., Claude 3.7, GPT‑4).
  - **Runtime & Deployment**: Built with Bun and deployed with CI/CD pipelines, integrating a CDN for serving generated documentation.

- **Folder Structure** (example):
site
    ├── bun.lock
    ├── components.json
    ├── data
    ├── docker-compose.dev.yml
    ├── Dockerfile
    ├── drizzle.config.ts
    ├── next.config.js
    ├── next-env.d.ts
    ├── package.json
    ├── postcss.config.js
    ├── prettier.config.js
    ├── public
    │   └── favicon.ico
    ├── README.md
    ├── src
    │   ├── app
    │   │   ├── api
    │   │   │   ├── auth
    │   │   │   │   └── [...nextauth]
    │   │   │   │       └── route.ts
    │   │   │   └── trpc
    │   │   │       └── [trpc]
    │   │   │           └── route.ts
    │   │   ├── _components
    │   │   │   └── post.tsx
    │   │   ├── layout.tsx
    │   │   └── page.tsx
    │   ├── components
    │   │   └── ui
    │   │       ├── button.tsx
    │   │       ├── form.tsx
    │   │       ├── hover-card.tsx
    │   │       ├── input.tsx
    │   │       ├── label.tsx
    │   │       ├── menubar.tsx
    │   │       ├── navigation-menu.tsx
    │   │       ├── popover.tsx
    │   │       ├── scroll-area.tsx
    │   │       ├── select.tsx
    │   │       ├── separator.tsx
    │   │       ├── sheet.tsx
    │   │       ├── sidebar.tsx
    │   │       ├── skeleton.tsx
    │   │       ├── slider.tsx
    │   │       ├── sonner.tsx
    │   │       ├── switch.tsx
    │   │       ├── toggle-group.tsx
    │   │       ├── toggle.tsx
    │   │       └── tooltip.tsx
    │   ├── env.js
    │   ├── hooks
    │   │   └── use-mobile.tsx
    │   ├── lib
    │   │   └── utils.ts
    │   ├── server
    │   │   ├── api
    │   │   │   ├── root.ts
    │   │   │   ├── routers
    │   │   │   │   └── post.ts
    │   │   │   └── trpc.ts
    │   │   ├── auth
    │   │   │   ├── config.ts
    │   │   │   └── index.ts
    │   │   └── db
    │   │       ├── index.ts
    │   │       └── schema.ts
    │   ├── styles
    │   │   └── globals.css
    │   └── trpc
    │       ├── query-client.ts
    │       ├── react.tsx
    │       └── server.ts
    ├── start-database.sh
    ├── tailwind.config.ts
    └── tsconfig.json

### 2. App Workflow

1. **Library Selection & Documentation Fetching**:
   - **User Flow**:
     - The user selects a library from popular registries (npm, PyPi, crates.io, composer, etc.) or uses a search function.
     - The selected library’s metadata (name, version, source URL) is fetched and displayed.
   - **Backend Process**:
     - An API call (via a TRPC endpoint) retrieves library details from your database or directly from registry APIs if not cached.

2. **Documentation Generation**:
   - The user chooses an LLM provider/model and optionally provides a custom API key.
   - Upon submission, the backend:
     1. Fetches the full documentation from the provided URL.
     2. Uses the Vercel AI-SDK to send the documentation text to the chosen LLM.
     3. Receives the condensed, LLM‑friendly version.
   - The generated documentation is stored in the database (supporting versioning) and tagged with metadata such as the provider/model used and optional sponsorship credits.

3. **Public Delivery and Versioning**:
   - Condensed docs are served as plain text from the CDN.
   - Multiple users can generate different versions for the same library, preserving a version history.

4. **Feedback and Rating**:
   - Users can rate and leave feedback on each generated document.
   - This data is used to improve future generations and help surface the most effective versions.

5. **Custom Documentation Upload**:
   - Users can also provide their own documentation files to be condensed, expanding the app’s use cases.

### 3. Data Model

use uuid for primary keys, no increments (for later optimizations if needed)
A proposed schema (using Drizzle ORM) includes the following entities:

- **User**:
  - `id` (PK)
  - `username`
  - `email`
  - `passwordHash` (if applicable)
  - `apiKeys` (for external LLM integration, json blob containing keys, find a way to secure if possible)
  - `createdAt`

- **Library**:
  - `id` (PK)
  - `name`
  - `source` (npm, pip, crates, composer,  etc...)
  - `latestVersion`
  - `documentationUrl`
  - `createdAt`

- **Documentation**:
  - `id` (PK)
  - `libraryId` (FK to Library)
  - `userId` (FK to User, if generated by a user)
  - `version` (version number or label)
  - `content` (plain text condensed documentation)
  - `provider` (e.g., Claude 3.7, GPT‑4)
  - `model` (model identifier)
  - `sponsoredBy` (optional credit)
  - `custom` (boolean to indicate if the docs were user‑provided)
  - `createdAt`
  - `public` (boolean)

- **Feedback**:
  - `id` (PK)
  - `documentationId` (FK to Documentation)
  - `userId` (FK to User)
  - `rating` (numeric, e.g., 1–5)
  - `feedbackText` (optional)
  - `createdAt`

### 4. Additional Functionality & Enhancements

To further improve the application in a practical way, consider the following enhancements:

1. **Real-Time Search and Filtering**:
   - Integrate a real‑time search bar that leverages TRPC endpoints to filter libraries by language, source, or popularity.
   - Implement auto‑completion to help users quickly find the desired library.

2. **Automated Documentation Updates**:
   - Set up background jobs to regularly check for new releases or updates on library registries.
   - When a new version is detected, automatically prompt users or generate an updated condensed version.

3. **Analytics and Usage Tracking**:
   - Collect and display statistics such as the number of generated documentations, user ratings, and feedback trends.
   - Use these insights to recommend popular libraries or highlight high‑quality documentation versions.

4. **Collaboration and Community Contributions**:
   - Allow users to suggest edits or improvements on generated documentation.
   - Introduce a community voting system to rank documentation quality and surface the best versions.

5. **Integration with Code Editors/IDEs**:
   - Develop plugins or extensions for popular IDEs (such as VS Code) so that developers can fetch condensed documentation on demand.
   - Leverage function‑calling mechanisms for seamless in‑editor documentation retrieval.

6. **Enhanced Security and API Key Management**:
   - Provide detailed documentation on how to securely store API keys.
   - Implement rate limiting and error handling to ensure reliable operation of TRPC endpoints and LLM calls.

7. **CDN Deployment and Caching**:
   - Deploy the generated documentation to a CDN to ensure low latency and high availability.
   - Use caching strategies to reduce repeated LLM calls and database queries.

8. **User Notifications & Subscriptions**:
   - Enable notifications (via email or in‑app alerts) when a library’s documentation is updated.
   - Allow users to subscribe to specific libraries or documentation topics for automatic updates.


## License

MIT License
