semantic_network = [
    ("Airline", "operates", "Flight"),
    ("Passenger", "books", "Flight"),
    ("Flight", "has", "Destination"),
    ("Flight", "has", "Departure Time"),
    # --- additional relationships ---
    ("Flight", "can be", "Delayed"),
    ("Flight", "can be", "Cancelled"),
    ("Delayed", "may cause", "Passenger Complaint"),
    ("Passenger", "files", "Passenger Complaint"),   # my own relationship
]

for relationship in semantic_network:
    print(relationship)
