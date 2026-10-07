package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/usehalo/halo/tui/internal/app"
	"github.com/usehalo/halo/tui/internal/render"
)

func main() {
	app.SetVersion(os.Getenv("HALO_VERSION"))
	render.DetectKittyGraphics()
	client, err := app.ConnectFromEnvironment()
	if err != nil {
		fmt.Fprintln(os.Stderr, "connect to Halo backend:", err)
		os.Exit(1)
	}
	defer client.Close()
	if err := client.Handshake(); err != nil {
		fmt.Fprintln(os.Stderr, "negotiate Halo TUI protocol:", err)
		os.Exit(1)
	}
	program := tea.NewProgram(app.New(client), tea.WithAltScreen(), tea.WithMouseCellMotion())
	finalModel, err := program.Run()
	if err != nil {
		fmt.Fprintln(os.Stderr, "run TUI:", err)
		os.Exit(1)
	}
	if model, ok := finalModel.(interface{ FatalError() error }); ok && model.FatalError() != nil {
		fmt.Fprintln(os.Stderr, "run TUI:", model.FatalError())
		os.Exit(1)
	}
}
