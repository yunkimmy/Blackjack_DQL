import random

class deck:
    def __init__(self):
        self.deck = self.make_deck()
        self.deck_length = len(self.deck)

    def __str__(self):
        deck_str = ', '.join(f'{rank} of {suit}' for suit, rank in self.deck)
        return f"Deck contains {self.deck_length} cards: {deck_str}"
    
    def restart(self):
        self.deck = self.make_deck()

    def draw_card(self):
        card = self.deck.pop()
        return card

    def make_deck(self):
        suits = ["Spades", "Hearts", "Diamonds", "Clubs"]
        
        deck = [(suit, rank) for suit in suits for rank in range(1,14)]
        random.shuffle(deck)

        return deck