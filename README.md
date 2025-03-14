# LLM-Docs

## Overview

LLM-Docs provides condensed, optimized documentation specifically tailored for efficient consumption by Large Language Models (LLMs). Traditional documentation, primarily designed for humans, often includes redundant content, extraneous formatting, and promotional material that unnecessarily occupies valuable context window space in LLMs. This project aims to create streamlined documentation, maximizing the efficiency and effectiveness of LLM-assisted programming tasks.

## Origin

This project originated from the following tweet:

[https://x.com/doodlestein/status/1900552731762253908](https://x.com/doodlestein/status/1900552731762253908)

> Just had an interesting idea for a useful service: go through all the python packages available from pip, in order of popularity.  
>  
> For each one, take the full, latest documentation website and feed it into Claude 3.7, section by section, asking it to put it into a much denser, plain-text form best suited for consumption by other LLMs for coding tasks.  
>  
> Then make a very simple website that serves these text files, like:  
>  
> llm-docs dot com slash pandas  
>  
> Then use function calling or MCP to make it easy for any model to grab these docs on demand as needed. The main insight here is that human docs have a ton of fluff that isn't really necessary and wastes context window space.  
>  
> And often it's also presented with a lot of formatting that makes it harder for the LLM to process, causing it to waste its cognition on silly, low-value stuff like fixing html tables that got mangled during copy/paste operations.  
>  
> I guess you'd want to do it for all languages, so you'd have:  
>  
> llm-docs dot com slash python slash pandas  
> llm-docs dot com slash js slash tailwindcss  
> llm-docs dot com slash rust slash serde  
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

- [`marqo_original_docs_merged.txt`](https://github.com/Dicklesworthstone/llm-docs/blob/main/marqo_original_docs_merged.txt): Original, full documentation from Marqo.
- [`marqo_condensed_docs_for_lllm.md`](https://github.com/Dicklesworthstone/llm-docs/blob/main/marqo_condensed_docs_for_lllm.md): Condensed, optimized version specifically crafted for LLM consumption.

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

> continue with part 1 of 5
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

## Future Directions

Potential future expansions include:

- Broadening library coverage based on popularity
- Extending documentation to other languages such as JavaScript, Rust, and Go
- Developing automated processing pipelines
- Establishing standardized formatting for consistency
- Offering documentation via a simplified API service
- Implementing version control for timely updates

## License

MIT License
