import json
from mtga import GameStateManager, MatchScanner, GameObject
from database import ArenaCardDatabase

class ActionParser:
    def __init__(self, card_db):
        self.game_state_manager = GameStateManager(card_db)
        self.previous_board_state = None

    def parse_log_line(self, log_line):
        state_changed = self.game_state_manager.parse_log_line(log_line)
        if state_changed:
            current_board_state = self.game_state_manager.get_current_board_state()
            if current_board_state:
                action_tokens = self.diff_board_states(self.previous_board_state, current_board_state)
                self.previous_board_state = current_board_state
                return action_tokens
        return []

    def diff_board_states(self, old_state, new_state):
        tokens = []
        if not old_state:
            tokens.append(f"GAME_START;match_id=1;player1_id={new_state.your_seat_id};player2_id={new_state.opponent_seat_id}")
            return tokens
        if old_state.current_turn != new_state.current_turn:
            tokens.append(f"TURN_CHANGE;player_id={new_state.your_seat_id if new_state.is_your_turn else new_state.opponent_seat_id};turn_number={new_state.current_turn}")
        if old_state.current_phase != new_state.current_phase:
            tokens.append(f"PHASE_CHANGE;phase_name={new_state.current_phase}")

        old_cards = {card.instance_id: card for card in self.get_all_cards(old_state)}
        new_cards = {card.instance_id: card for card in self.get_all_cards(new_state)}

        for card_id, new_card in new_cards.items():
            if card_id not in old_cards:
                tokens.append(f"ZONE_TRANSFER;card_instance_id={card_id};grp_id={new_card.grp_id};source_zone=None;destination_zone={new_card.zone_id}")
            else:
                old_card = old_cards[card_id]
                if old_card.zone_id != new_card.zone_id:
                    tokens.append(f"ZONE_TRANSFER;card_instance_id={card_id};grp_id={new_card.grp_id};source_zone={old_card.zone_id};destination_zone={new_card.zone_id}")

        old_attackers = {card.instance_id for card in old_state.your_battlefield if card.is_attacking}
        new_attackers = {card.instance_id for card in new_state.your_battlefield if card.is_attacking}
        declared_attackers = new_attackers - old_attackers
        for attacker_id in declared_attackers:
            tokens.append(f"DECLARE_ATTACKER;attacker_id={attacker_id};target_id={new_state.opponent_seat_id}")

        if old_state.opponent_life > new_state.opponent_life:
            damage = old_state.opponent_life - new_state.opponent_life
            tokens.append(f"DAMAGE_DEALT;source_id=player;target_id={new_state.opponent_seat_id};amount={damage}")

        return tokens

    def get_all_cards(self, board_state):
        if not board_state: return []
        return (board_state.your_hand + board_state.your_battlefield + board_state.your_graveyard +
                board_state.opponent_battlefield + board_state.opponent_graveyard)

class StateReconstructor:
    def __init__(self, card_db):
        self.card_db = card_db
        self.game_state_manager = GameStateManager(self.card_db)

    def apply_tokens(self, tokens):
        for token in tokens:
            parts = token.split(';')
            action_type = parts[0]
            params = dict(p.split('=') for p in parts[1:])

            handler = getattr(self, f"_handle_{action_type.lower()}", None)
            if handler:
                handler(params)
        return self.game_state_manager.get_current_board_state()

    def _handle_game_start(self, params):
        self.game_state_manager.scanner.local_player_seat_id = int(params['player1_id'])

    def _handle_turn_change(self, params):
        self.game_state_manager.scanner.current_turn = int(params['turn_number'])
        self.game_state_manager.scanner.active_player_seat = int(params['player_id'])

    def _handle_phase_change(self, params):
        self.game_state_manager.scanner.current_phase = params['phase_name']

    def _handle_zone_transfer(self, params):
        instance_id = int(params['card_instance_id'])
        grp_id = int(params['grp_id'])
        dest_zone = int(params['destination_zone']) if params['destination_zone'] != 'None' else 0

        if instance_id not in self.game_state_manager.scanner.game_objects:
            # A simplified object creation. A real implementation needs more details.
            self.game_state_manager.scanner.game_objects[instance_id] = GameObject(
                instance_id=instance_id, grp_id=grp_id, zone_id=dest_zone, owner_seat_id=1 # Assuming owner is player 1
            )
        else:
            self.game_state_manager.scanner.game_objects[instance_id].zone_id = dest_zone

if __name__ == '__main__':
    card_db = ArenaCardDatabase()
    parser = ActionParser(card_db)
    reconstructor = StateReconstructor(card_db)

    print("ActionParser and StateReconstructor created. Ready to process logs.")

    # This example demonstrates the full loop: logs -> tokens -> state
    sample_tokens = [
        "GAME_START;match_id=1;player1_id=1;player2_id=2",
        "TURN_CHANGE;player_id=1;turn_number=1",
        "PHASE_CHANGE;phase_name=Main",
        "ZONE_TRANSFER;card_instance_id=123;grp_id=67352;source_zone=None;destination_zone=2" # Card to battlefield
    ]

    final_state = reconstructor.apply_tokens(sample_tokens)
    print("\nReconstructed State:")
    print(f"Turn: {final_state.current_turn}, Phase: {final_state.current_phase}")
    print(f"Cards on battlefield: {len(final_state.your_battlefield)}")
