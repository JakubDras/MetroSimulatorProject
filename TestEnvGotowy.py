import gymnasium as gym
import numpy as np
import pygame
import sys
import random


from MMEnvKuba import MiniMetroEnv, Station, Line, Train, euclidean_distance, \
    draw_passenger_icon


first_station_clicked_idx = None


def map_mouse_to_action(env, mouse_pos, event_type):

    action = {
        "high_level_action": 0,
        "low_level_params": np.array([0, 0, 0], dtype=np.int32)
    }

    global first_station_clicked_idx

    if event_type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:
            clicked_station_idx = None

            for i, station in enumerate(env.stations):

                if hasattr(station, 'get_rect') and station.get_rect().collidepoint(mouse_pos):
                    clicked_station_idx = i
                    break

            if env.deploy_train_mode:
                if clicked_station_idx is not None and env.available_trains > 0:

                    station_on_eligible_line = False
                    for line in env.lines:

                        if any(s.station_id == env.stations[clicked_station_idx].station_id for s in
                               line.stations) and len(line.stations) >= 2:
                            station_on_eligible_line = True
                            break

                    if station_on_eligible_line:
                        action["high_level_action"] = 2
                        action["low_level_params"][0] = clicked_station_idx
                        print(f"Akcja: Rozmieść pociąg na stacji {env.stations[clicked_station_idx].station_id}")
                    else:
                        print(
                            "Nie można rozmieścić pociągu: Stacja nie należy do linii z min. 2 stacjami lub brak wolnych pociągów.")
                else:
                    print("Nie można rozmieścić pociągu: Brak stacji lub dostępnych pociągów.")
                env.deploy_train_mode = False
                first_station_clicked_idx = None
            elif clicked_station_idx is not None:
                if first_station_clicked_idx is None:
                    first_station_clicked_idx = clicked_station_idx
                    print(f"Wybrano stację początkową: {env.stations[first_station_clicked_idx].station_id}")
                else:
                    station1_idx = first_station_clicked_idx
                    station2_idx = clicked_station_idx

                    if station1_idx != station2_idx:

                        if env.selected_line_index is not None:
                            action["high_level_action"] = 1
                            action["low_level_params"][0] = 0
                            action["low_level_params"][1] = station1_idx
                            action["low_level_params"][2] = station2_idx
                            print(
                                f"Akcja: Budowanie/Przedłużanie linii między {env.stations[station1_idx].station_id} i {env.stations[station2_idx].station_id}")
                        else:
                            print("Nie wybrano koloru linii. Użyj klawiszy 1-9.")
                    else:
                        print("Wybrano tę samą stację. Akcja pominięta.")
                    first_station_clicked_idx = None
            else:
                first_station_clicked_idx = None
                env.deploy_train_mode = False
                print("Kliknięto w puste miejsce. Reset stacji początkowej.")


        elif event.button == 3:
            if env.selected_line_index is not None and len(env.lines) > 0:
                current_line_color_to_remove = env.LINE_COLORS[env.selected_line_index]

                if any(l.color == current_line_color_to_remove for l in env.lines):
                    action["high_level_action"] = 1
                    action["low_level_params"][0] = 1
                    action["low_level_params"][1] = 0
                    action["low_level_params"][2] = 0
                    print(f"Akcja: Próba usunięcia linii koloru: {current_line_color_to_remove}")
                else:
                    print(f"Brak aktywnej linii o kolorze {current_line_color_to_remove} do usunięcia. Robię NOOP.")
            else:
                print("Brak aktywnej linii do usunięcia lub brak wybranych linii. Robię NOOP.")
            first_station_clicked_idx = None

    elif event_type == pygame.KEYDOWN:

        if pygame.K_1 <= event.key <= pygame.K_9:
            line_color_idx = event.key - pygame.K_1
            if line_color_idx < len(env.LINE_COLORS):
                action["high_level_action"] = 3
                action["low_level_params"][0] = line_color_idx
                print(f"Akcja: Wybrano linię o kolorze: {env.LINE_COLORS[line_color_idx]}")
            else:
                print(
                    f"Kolor linii o indeksie {line_color_idx} nie istnieje. Maksymalnie {len(env.LINE_COLORS)} kolorów.")

        elif event.key == pygame.K_SPACE:
            env.deploy_train_mode = not env.deploy_train_mode
            print(
                f"Tryb rozmieszczania pociągu: {'Włączony' if env.deploy_train_mode else 'Wyłączony'}. Kliknij stację, aby rozmieścić.")
            first_station_clicked_idx = None

        elif event.key == pygame.K_RETURN:
            action["high_level_action"] = 4
            print("Akcja: Zakończono tydzień ręcznie.")
            first_station_clicked_idx = None

    return action


# --- GŁÓWNA PĘTLA GRY ---
if __name__ == "__main__":
    env = MiniMetroEnv(
        render_mode="human",
    )

    obs, info = env.reset()
    terminated = False
    truncated = False

    print("Gra zresetowana. Naciśnij dowolny klawisz, aby rozpocząć interakcję...")
    print("Sterowanie:")
    print("  Lewy klik myszy: Wybierz stację (pierwsze kliknięcie), połącz z drugą stacją (drugie kliknięcie)")
    print("  Prawy klik myszy: Usuń aktywną linię")
    print("  Klawisze 1-9: Wybierz kolor linii (aktywną linię)")
    print("  SPACJA: Przełącz tryb rozmieszczania pociągu (po włączeniu kliknij stację, aby rozmieścić)")
    print("  ENTER: Zakończ bieżący tydzień")
    print("  ESC / Zamknij okno: Wyjście z gry")

    # --- Pauza startowa ---
    env.render()

    waiting_to_start = True
    while waiting_to_start:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                waiting_to_start = False
            if event.type == pygame.KEYDOWN:
                waiting_to_start = False
        pygame.display.flip()
        if hasattr(env, 'clock') and env.metadata.get("render_fps"):
            env.clock.tick(env.metadata["render_fps"])
        else:
            pygame.time.Clock().tick(30)
    # --- Koniec pauzy startowej ---

    running = True

    FPS = env.metadata.get("render_fps", 60)

    while running:
        current_action = {
            "high_level_action": 0,  # Domyślnie NOOP
            "low_level_params": np.array([0, 0, 0], dtype=np.int32)
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            action_from_input = map_mouse_to_action(env, pygame.mouse.get_pos(), event.type)
            if action_from_input["high_level_action"] != 0:
                current_action = action_from_input

        obs, reward, terminated, truncated, info = env.step(current_action)

        env.render()

        if terminated or truncated:
            print(f"Gra zakończona! Wynik: {info['score']}, Tydzień: {info['week_number']}")
            running = False

        if hasattr(env, 'clock'):
            env.clock.tick(FPS)
        else:
            pygame.time.Clock().tick(FPS)

    env.close()
    pygame.quit()
    sys.exit()