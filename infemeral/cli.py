"""CLI tool for managing output styles and Cursor commands."""

import re
import sys
from pathlib import Path
from typing import NamedTuple


class OutputStyle(NamedTuple):
    """Represents an output style configuration."""

    name: str
    description: str
    filename: str
    slug: str  # filename without .md extension


def scan_output_styles(claude_dir: Path | None = None) -> list[OutputStyle]:
    """Scan .claude/output-styles/ directory for available output styles.

    Args:
        claude_dir: Path to .claude directory (defaults to project root)

    Returns:
        List of OutputStyle objects sorted by name
    """
    if claude_dir is None:
        # Try to find .claude directory relative to this file
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        claude_dir = project_root / ".claude"
    else:
        claude_dir = Path(claude_dir)

    styles_dir = claude_dir / "output-styles"
    if not styles_dir.exists():
        return []

    styles = []
    for style_file in sorted(styles_dir.glob("*.md")):
        try:
            content = style_file.read_text()
            # Parse YAML frontmatter
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if not match:
                continue

            frontmatter = match.group(1)
            name_match = re.search(r"^name:\s*(.+)$", frontmatter, re.MULTILINE)
            desc_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)

            name = name_match.group(1).strip() if name_match else style_file.stem
            description = desc_match.group(1).strip() if desc_match else ""

            # Create slug from filename (without .md)
            slug = style_file.stem

            styles.append(OutputStyle(
                name=name,
                description=description,
                filename=style_file.name,
                slug=slug,
            ))
        except Exception:
            # Skip files that can't be parsed
            continue

    return sorted(styles, key=lambda s: s.name.lower())


def show_output_styles_menu(styles: list[OutputStyle]) -> OutputStyle | None:
    """Display an interactive menu for selecting an output style.

    Args:
        styles: List of available output styles

    Returns:
        Selected OutputStyle or None if cancelled
    """
    if not styles:
        print("No output styles found.", file=sys.stderr)
        return None

    print("\nAvailable Output Styles:")
    print("=" * 60)

    # Display numbered menu
    for idx, style in enumerate(styles, start=1):
        print(f"{idx:2d}. {style.name:<30} - {style.description}")

    print("=" * 60)
    print("0. Cancel")

    while True:
        try:
            choice = input("\nSelect an output style (number): ").strip()
            if choice == "0":
                return None

            idx = int(choice)
            if 1 <= idx <= len(styles):
                return styles[idx - 1]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(styles)}.", file=sys.stderr)
        except ValueError:
            print("Invalid input. Please enter a number.", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return None


def list_output_styles(styles: list[OutputStyle], format: str = "text") -> None:
    """List available output styles in the specified format.

    Args:
        styles: List of available output styles
        format: Output format ('text', 'json', 'yaml')
    """
    if format == "json":
        import json
        output = json.dumps([
            {
                "name": style.name,
                "description": style.description,
                "slug": style.slug,
                "filename": style.filename,
            }
            for style in styles
        ], indent=2)
        print(output)
    elif format == "yaml":
        try:
            import yaml
            output = yaml.dump([
                {
                    "name": style.name,
                    "description": style.description,
                    "slug": style.slug,
                    "filename": style.filename,
                }
                for style in styles
            ], default_flow_style=False, sort_keys=False)
            print(output)
        except ImportError:
            print("Error: PyYAML is required for YAML output. Install with: pip install pyyaml", file=sys.stderr)
            sys.exit(1)
    else:  # text format
        if not styles:
            print("No output styles found.")
            return

        print("\nAvailable Output Styles:")
        print("=" * 80)
        print(f"{'Name':<30} {'Description':<40} {'Slug':<10}")
        print("-" * 80)
        for style in styles:
            print(f"{style.name:<30} {style.description:<40} {style.slug:<10}")
        print("=" * 80)
        print(f"\nTotal: {len(styles)} output styles")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage and select output styles for Cursor commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive menu to select output style
  %(prog)s --list             # List all available output styles
  %(prog)s --list --format json  # List output styles in JSON format
  %(prog)s --select json-structured  # Select a specific style by slug
        """,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available output styles",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format for --list (default: text)",
    )
    parser.add_argument(
        "--select",
        metavar="SLUG",
        help="Select a specific output style by slug (e.g., 'json-structured')",
    )
    parser.add_argument(
        "--claude-dir",
        type=Path,
        help="Path to .claude directory (defaults to project root)",
    )

    args = parser.parse_args()

    # Scan for output styles
    styles = scan_output_styles(args.claude_dir)

    if args.list:
        list_output_styles(styles, format=args.format)
        return

    if args.select:
        # Find style by slug
        selected = next((s for s in styles if s.slug == args.select), None)
        if selected:
            print(selected.slug)
            return 0
        else:
            print(f"Error: Output style '{args.select}' not found.", file=sys.stderr)
            print("\nAvailable slugs:", file=sys.stderr)
            for style in styles:
                print(f"  - {style.slug}", file=sys.stderr)
            return 1

    # Interactive menu mode
    selected = show_output_styles_menu(styles)
    if selected:
        print(selected.slug)
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
