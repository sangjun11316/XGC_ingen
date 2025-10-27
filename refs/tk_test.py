import tkinter as tk

# 1. Create the main application window
root = tk.Tk()
root.title("Hello Tkinter") # Set the window title

# 2. Create a label widget
label = tk.Label(root, text="Hello, World!")

# 3. Add the label to the window
# The pack() method is a simple way to place widgets
label.pack(padx=20, pady=20) # Add some padding

# 4. Start the Tkinter event loop
# This keeps the window open and responsive
root.mainloop()
