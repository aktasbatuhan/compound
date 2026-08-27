# Compound README demo — rendered with charmbracelet/vhs
#   vhs assets/cli-demo.tape   ->   assets/cli-demo.gif
# On-brand paper theme (matches compound-1js.pages.dev). Every command is real
# and free: `list` enumerates the benchmark library, `providers` enumerates the
# hosts serving one model with their quant, price, and live status.

Output assets/cli-demo.gif

Set Shell "bash"
Set FontFamily "Geist Mono"
Set FontSize 18
Set LineHeight 1.3
Set Width 1500
Set Height 840
Set Padding 28
Set Margin 36
Set MarginFill "#e9e4d8"
Set WindowBar Colorful
Set WindowBarSize 44
Set BorderRadius 12
Set TypingSpeed 38ms

# Paper theme: warm ink on the site's #faf9f5, brand-blue accents.
Set Theme { "background": "#faf9f5", "foreground": "#2a251d", "cursor": "#1740e6", "selection": "#d9e0fb", "black": "#3a352c", "red": "#b23a2e", "green": "#3f7d3a", "yellow": "#9a6a15", "blue": "#1740e6", "magenta": "#8a3ec0", "cyan": "#1f7a80", "white": "#2a251d", "brightBlack": "#8c857a", "brightRed": "#b23a2e", "brightGreen": "#3f7d3a", "brightYellow": "#9a6a15", "brightBlue": "#1740e6", "brightMagenta": "#8a3ec0", "brightCyan": "#1f7a80", "brightWhite": "#2a251d" }

Hide
Type "source .venv/bin/activate" Enter
Type `export PS1='\[\e[38;2;23;64;230m\]$\[\e[0m\] '` Enter
Type "clear" Enter
Show

Sleep 700ms

# Beat 1 — one front door to five benchmarks.
Type "compound-bench list"
Sleep 500ms
Enter
Sleep 3.2s

Type "clear" Enter
Sleep 400ms

# Beat 2 — one model, every host serving it: quant, price, live status.
Type "compound-bench providers deepseek/deepseek-v4-flash-0731 | head -18"
Sleep 500ms
Enter
Sleep 4.5s

Sleep 1.5s
