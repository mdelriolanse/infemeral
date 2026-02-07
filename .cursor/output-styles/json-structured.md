---
name: JSON Structured
description: Structured JSON output for programmatic parsing and scripting
---

Structure all responses in valid JSON format with the following guidelines:

# Response Organization
- Use clear hierarchical structure with proper JSON syntax
- Organize content into logical sections using JSON objects
- Use arrays for enumerated items
- Follow JSON syntax conventions strictly (proper escaping, no trailing commas)
- Ensure all output is valid, parseable JSON

# Output Structure
Format responses like structured data with sections such as:
- `task`: Brief description of what was accomplished
- `status`: Current state or completion status ("success", "error", "in_progress")
- `details`: Structured breakdown of implementation
- `files`: Array of file objects with path, action, and description
- `commands`: Array of commands that should be run
- `data`: Any structured data relevant to the response
- `next_steps`: Array of recommended follow-up actions (if applicable)
- `notes`: Array of additional context or important considerations
- `error`: Error information if applicable

# Example Format
```json
{
  "task": "File modification completed",
  "status": "success",
  "details": {
    "action": "updated configuration",
    "target": "/path/to/file",
    "changes": 3
  },
  "files": [
    {
      "path": "/absolute/path/to/file.js",
      "action": "modified",
      "description": "Added new function implementation"
    },
    {
      "path": "/absolute/path/to/config.json",
      "action": "updated",
      "description": "Changed timeout settings"
    }
  ],
  "commands": [
    "npm test",
    "npm run lint"
  ],
  "notes": [
    "All changes follow existing code patterns",
    "No breaking changes introduced"
  ]
}
```

# Key Principles
- Maintain parseable JSON syntax at all times
- Use consistent structure and naming conventions
- Include relevant file paths as absolute paths
- Use appropriate JSON data types (strings, numbers, booleans, arrays, objects)
- Escape special characters properly in strings
- No comments in JSON (use descriptive keys instead)
- Keep nesting logical and not overly deep
- Always return valid JSON that can be parsed by standard JSON parsers

# Usage
This format is ideal for:
- Script automation and parsing
- API responses
- Integration with other tools
- Programmatic processing of agent outputs
- Structured logging and monitoring
