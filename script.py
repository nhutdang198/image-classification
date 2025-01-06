from rich.console import Console
from rich.prompt import Prompt

console = Console()

def menu():
    options = ["Train", "Validation", "Exit"]
    while True:
        console.print("Image classification:")
        for idx, option in enumerate(options):
            console.print(f"[bold cyan]{idx + 1}[/] - {option}")
        choice = Prompt.ask("Select an option", choices=[str(i+1) for i in range(len(options))], default="1")
        if choice == "3":
            console.print("[bold green]Exiting...[/]")
            break
        else:
            console.print(f"You selected: [bold yellow]{options[int(choice)-1]}[/]")

menu()
