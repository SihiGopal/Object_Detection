import tkinter as tk
from tkinter import messagebox

# Caesar Cipher Encrypt function
def encrypt_message():
    message = entry_message.get()
    shift = int(entry_shift.get())
    encrypted_text = "".join(
        chr((ord(char) - 65 + shift) % 26 + 65) if char.isupper() else
        chr((ord(char) - 97 + shift) % 26 + 97) if char.islower() else char
        for char in message
    )
    entry_result.delete(0, tk.END)
    entry_result.insert(0, encrypted_text)

# Caesar Cipher Decrypt function
def decrypt_message():
    message = entry_message.get()
    shift = int(entry_shift.get())
    decrypted_text = "".join(
        chr((ord(char) - 65 - shift) % 26 + 65) if char.isupper() else
        chr((ord(char) - 97 - shift) % 26 + 97) if char.islower() else char
        for char in message
    )
    entry_result.delete(0, tk.END)
    entry_result.insert(0, decrypted_text)

# Tkinter GUI setup
root = tk.Tk()
root.title("Secret Message Encryption and Decryption Tool")

# Message Label and Entry
tk.Label(root, text="Enter Message:").grid(row=0, column=0, padx=10, pady=10)
entry_message = tk.Entry(root, width=40)
entry_message.grid(row=0, column=1, padx=10, pady=10)

# Shift Label and Entry
tk.Label(root, text="Enter Shift (key):").grid(row=1, column=0, padx=10, pady=10)
entry_shift = tk.Entry(root, width=40)
entry_shift.grid(row=1, column=1, padx=10, pady=10)

# Result Label and Entry
tk.Label(root, text="Result:").grid(row=2, column=0, padx=10, pady=10)
entry_result = tk.Entry(root, width=40)
entry_result.grid(row=2, column=1, padx=10, pady=10)

# Encrypt and Decrypt Buttons
btn_encrypt = tk.Button(root, text="Encrypt", command=encrypt_message)
btn_encrypt.grid(row=3, column=0, padx=10, pady=10)

btn_decrypt = tk.Button(root, text="Decrypt", command=decrypt_message)
btn_decrypt.grid(row=3, column=1, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
