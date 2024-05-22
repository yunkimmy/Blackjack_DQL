from deck import deck
from typing import Tuple
import torch
#### action -> hit: 0, stay: 1 ######
#### turn -> player side: 0, dealer side: 1 ######
#### state = {"dealer" : {"ace" : int, "hand": list},"player" : {"ace" : int, "hand": list}, "turn": int} ########

class blackjack:
    def __init__(self):
        self.deck = deck()
        self.hidden_card = None

    def reset(self):
        self.deck.restart()

        keys = ["ace", "hand"]
        dealer_values = [0, []]
        player_values = [0, []]
        dealer = dict(zip(keys, dealer_values))
        player = dict(zip(keys, player_values))

        for _ in range(2):
            dealer["hand"].append(self.deck.draw_card())
            player["hand"].append(self.deck.draw_card())
        self.count_ace(dealer)
        self.count_ace(player)

        self.hidden_card = dealer["hand"].pop()

        state = {"dealer":dealer, "player":player, "turn":0}

        return state
        
    def count_ace(self, hand: dict) -> None:
        sum_ace = 0
        for suit, rank in hand["hand"]:
            if rank == 1:
                sum_ace +=1 
        hand["ace"] = sum_ace
            
    def is_ten(self, card_num: int) -> bool:
        ten = [10, 11, 12, 13]
        return card_num in ten

    def get_sum(self, hand: dict) -> int:
        sum = 0
        num_ace = hand["ace"]
        for suit, rank in hand["hand"]:
            if self.is_ten(rank):   # if card is 10,J,Q,K
                sum += 10
            elif rank == 1:         # if card is A
                sum += 11
            else:
                sum += rank

        if sum > 21:   # bust
            for _ in range(num_ace):
                sum -= 10           # change the aces to ones
                if sum < 22:        
                    break

        return sum

    def is_bust(self, hand: dict) -> bool:
        sum = self.get_sum(hand)
        if sum > 21:
            return True
        else:
            return False
    
    def count_cards(self, hand: dict) -> bool:    #check if number of cards is 5
        length = len(hand["hand"])
        if length > 4:
            return True
        else:
            return False
    
    def is_terminal(self, state: dict) -> bool:
        dealer = state["dealer"]
        player = state["player"]
        turn = state["turn"]

        if turn:   # dealer side
            if self.count_cards(dealer):        # number of cards is 5
                return True
            
            sum_hand = self.get_sum(dealer)
            if sum_hand < 17: # keep hitting
                return False
            else:               # bust or stand            
                return True
            
        else:
            sum_hand = self.get_sum(player)
            if sum_hand < 22:
                return False
            else:               # bust
                return True

    
    def get_reward(self, state: dict) -> int:
        dealer = state["dealer"]
        player = state["player"]
        turn = state["turn"]
        if self.is_terminal(state): # game is over
            if turn == 0:
                reward = -1         # player bust
            elif turn == 1:
                dealer_sum = self.get_sum(dealer)
                player_sum = self.get_sum(player)
                if dealer_sum > 21 or player_sum > dealer_sum:      # dealer bust or player has more
                    reward = 1
                elif dealer_sum > player_sum:       # dealer won
                    reward = -1
                else:
                    dealer_length = len(dealer["hand"])
                    player_length = len(player["hand"])
                    if dealer_length > player_length:
                        reward = -1
                    elif player_length > dealer_length:
                        reward = 1
                    else:
                        reward = 0
        else:
            reward = 0      # game is till playing

        return torch.tensor(reward, dtype = torch.float32)
    
    def step(self, state: dict, action: int) -> Tuple[dict, int, bool]:
        turn = state["turn"]

        if turn == 0:  # player's turn
            next_state, reward, done = self.player_interaction(state, action)
        else:
            next_state, reward, done = self.dealer_interaction(state)

        return next_state, reward, done

    def player_interaction(self, state: dict, action: int) -> Tuple[dict, int, bool]:
        dealer = state["dealer"]
        player = state["player"]
        turn = state["turn"]
        

        if action:   # if you stand, the turn goes to the dealer
            turn = 1
        else:
            player["hand"].append(self.deck.draw_card())
            self.count_ace(player)

        if self.count_cards(player):
            turn = 1
        
        if turn:
            dealer["hand"].append(self.hidden_card)

        next_state = {"dealer" : dealer, "player" : player, "turn" : turn}

        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)

        return next_state, reward, done
    
    def dealer_interaction(self, state:dict) -> Tuple[dict, int, bool]:
        dealer = state["dealer"]
        player = state["player"]
        turn = state["turn"]

        while not self.is_terminal(state):     # have to hit if sum is below 17
            dealer["hand"].append(self.deck.draw_card())            # get card
            self.count_ace(dealer)                                  # update aces
        
        next_state = {"dealer" : dealer, "player" : player, "turn" : turn}

        reward = self.get_reward(next_state)

        return next_state , reward, True

