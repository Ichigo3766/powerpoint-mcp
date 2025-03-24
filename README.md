# powerpoint MCP server

A MCP server project that creates powerpoint presentations

## Components

### Tools

The server implements multiple tools:
- ```create-presentation```: Starts a presentation
  - Takes "name"  as required string arguments
  - Creates a presentation object
- ```add-slide-title-only```: Adds a title slide to the presentation
  - Takes "presentation_name" and "title" as required string arguments
  - Creates a title slide with "title" and adds it to presentation
- ```add-slide-section-header```: Adds a section header slide to the presentation
  - Takes "presentation_name" and "header" as required string arguments
  - Creates a section header slide with "header" (and optionally "subtitle") and adds it to the presentation
- ```add-slide-title-content```: Adds a title with content slide to the presentation
  - Takes "presentation_name", "title", "content" as required string arguments
  - Creates a title with content slide with "title" and "content" and adds it to presentation
- ```add-slide-title-with-table```: Adds a title slide with a table
  - Takes "presentation_name", "title", "data" as required string and array arguments
  - Creates a title slide with "title" and adds a table dynamically built from data
- ```add-slide-title-with-chart```: Adds a title slide with a chart
  - Takes "presentation_name", "title", "data" as required string and object arguments
  - Creates a title slide with "title" and adds a chart dynamically built from data. Attempts to figure out the best type of chart from the data source.
- ```add-slide-picture-with-caption```: Adds a picture with caption slide
  - Takes "presentation_name", "title", "caption", "image_path" as required string arguments
  - Creates a picture with caption slide using the supplied "title", "caption", and "image_path". Can either use images created via the "generate-and-save-image" tool or use an "image_path" supplied by the user (image must exist in folder_path)
- ```open-presentation```: Opens a presentation for editing
  - Takes "presentation_name" as required arguments
  - Opens the given presentation and automatically saves a backup of it as "backup.pptx"
  - This tool allows the client to work with existing pptx files and add slides to them. Just make sure the client calls "save-presentation" tool at the end.
- ```save-presentation```: Saves the presentation to a file.
  - Takes "presentation_name" as required arguments.
  - Saves the presentation to the folder_path. The client must call this tool to finalize the process.
- ```generate-and-save-image```: Generates an image for the presentation using a T2I model
  - Takes "prompt" and "file_name" as required string arguments
  - Creates an image using stable diffusion api (ForgeUI/Automatic1111)

## Configuration

An environment variable is required for image generation via ForgeUI API

```
"env": 
{
  "SD_WEBUI_URL": "http://your-sd-webui-url:7860",
  "SD_AUTH_USER": "your-username",  // Optional: if authentication is enabled
  "SD_AUTH_PASS": "your-password",  // Optional: if authentication is enabled
}
```

A folder_path is required. All presentations and images will be saved to this folder.

```
"--folder-path",
  "/path/to/decks_folder"
```

## Quickstart

### Install

#### Make sure you have UV installed

MacOS/Linux
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Clone the repo

```
git clone https://github.com/Ichigo3766/powerpoint-mcp.git
```

#### Claude Desktop (works with any mcp client)

On MacOS: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

- ```--directory```: the path where you cloned the repo above (ex: /full/path/to/powerpoint-mcp/src)
- ```--folder-path```: the path where powerpoint decks and images will be saved to. Also the path where you should place any images you want the MCP server to use.

```
  # Add the server to your claude_desktop_config.json
  "mcpServers": {
    "powerpoint": {
      "command": "uv",
      "env": {
        "SD_WEBUI_URL": "http://your-sd-webui-url:7860",
        "SD_AUTH_USER": "your-username",  // Optional: if authentication is enabled
        "SD_AUTH_PASS": "your-password",  // Optional: if authentication is enabled
      },
      "args": [
        "--directory",
        "/path/to/powerpoint",
        "run",
        "powerpoint",
        "--folder-path",
        "/path/to/decks_folder"
      ]
    }
  }
```

### Usage Examples

```
Create a presentation about fish, create some images and include tables and charts
```

```
Create a presentation about the attached paper. Please use the following images in the presentation:
author.jpeg
```

Assuming you have SQLite MCP Server installed.
```
Review 2024 Sales Data table. Create a presentation showing current trends, use tables and charts as appropriate
```

## Credits

This project is a fork of [powerpoint](https://github.com/supercurses/powerpoint) by [supercurses](https://github.com/supercurses), with additional features and modifications. The original project provided the foundation for this MCP server.

## License

This MCP server is licensed under the MIT License. This means you are free to use, modify, and distribute the software, subject to the terms and conditions of the MIT License. For more details, please see the LICENSE file in the project repository.