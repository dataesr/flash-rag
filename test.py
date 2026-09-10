import unicodedata

a = "les activit\u00e9s - de R&D (dans )les biotechnologies"
b = "".join(c if c.isalnum() else " " for c in unicodedata.normalize("NFD", a.lower()) if unicodedata.category(c) != "Mn")
d = "_".join(b.split())
print(d)
