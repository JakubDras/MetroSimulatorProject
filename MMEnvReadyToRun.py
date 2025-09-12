import pygame
import sys
import numpy as np
from MMEnvKuba import MiniMetroEnv


first_station_clicked_idx = None

def create_noop_action():
    return {
        "high_level_action": 0,
        "low_level_params": np.array([0, 0, 0], dtype=np.int32)
    }

def map_mouse_to_action(env, event):
    global first_station_clicked_idx
    action = create_noop_action()
    mouse_pos = pygame.mouse.get_pos()

    # --- Obsługa zdarzeń myszy ---
    if event.type == pygame.MOUSEBUTTONDOWN:
        # --- LEWY PRZYCISK MYSZY ---
        if event.button == 1:
            for i, rect in enumerate(env.ui_circles_rects):
                if rect.collidepoint(mouse_pos):
                    action["high_level_action"] = 3
                    action["low_level_params"][1] = i
                    print(f"Akcja UI: Wybrano linię o indeksie {i}")
                    return action

            # Sprawdzenie przycisku "Train"
            if env.TRAIN_BUTTON_RECT.collidepoint(mouse_pos):
                env.deploy_train_mode = not env.deploy_train_mode
                print(f"Tryb rozmieszczania pociągu: {'Włączony' if env.deploy_train_mode else 'Wyłączony'}")
                return create_noop_action()

            # 2. Jeśli nie kliknięto w UI, sprawdź stacje
            clicked_station_idx = None
            for i, station in enumerate(env.stations):
                if station.is_clicked(mouse_pos):
                    clicked_station_idx = i
                    break

            # 3. Logika akcji w zależności od trybu i kliknięcia
            if env.deploy_train_mode:
                if clicked_station_idx is not None:
                    action["high_level_action"] = 2
                    action["low_level_params"][1] = clicked_station_idx
                    print(f"Akcja: Rozmieść pociąg na stacji o indeksie {clicked_station_idx}")
                else:
                    print("Anulowano tryb rozmieszczania pociągu (kliknięto w puste miejsce).")
                env.deploy_train_mode = False
                first_station_clicked_idx = None
            # Tryb budowania linii
            elif clicked_station_idx is not None:
                if first_station_clicked_idx is None:
                    first_station_clicked_idx = clicked_station_idx
                    print(f"Wybrano stację początkową: {env.stations[first_station_clicked_idx].station_id}")
                else:
                    if first_station_clicked_idx != clicked_station_idx:
                        action["high_level_action"] = 1
                        action["low_level_params"][0] = 0
                        action["low_level_params"][1] = first_station_clicked_idx
                        action["low_level_params"][2] = clicked_station_idx
                        print(
                            f"Akcja: Buduj/Przedłuż linię między stacjami o indeksach {first_station_clicked_idx} i {clicked_station_idx}")
                    else:
                        print("Anulowano wybór (kliknięto tę samą stację).")
                    first_station_clicked_idx = None
            else:
                print("Anulowano wybór (kliknięto w puste miejsce).")
                first_station_clicked_idx = None

        # --- PRAWY PRZYCISK MYSZY ---
        elif event.button == 3:
            current_line_color = env.LINE_COLORS[env.selected_line_index]
            if any(line.color == current_line_color for line in env.lines):
                action["high_level_action"] = 1
                action["low_level_params"][0] = 1
                print(f"Akcja: Usuń linię o kolorze {current_line_color}")
            else:
                print(f"Brak linii o aktywnym kolorze do usunięcia.")
            first_station_clicked_idx = None
    # --- Obsługa zdarzeń klawiatury ---
    elif event.type == pygame.KEYDOWN:
        # Wybór koloru linii klawiszami 1-6
        if pygame.K_1 <= event.key <= pygame.K_6:
            line_idx = event.key - pygame.K_1
            if line_idx < len(env.LINE_COLORS):
                action["high_level_action"] = 3
                action["low_level_params"][1] = line_idx
                print(f"Akcja: Wybrano linię o indeksie {line_idx}")
        # Przełączanie trybu rozmieszczania pociągu
        elif event.key == pygame.K_SPACE:
            env.deploy_train_mode = not env.deploy_train_mode
            print(f"Tryb rozmieszczania pociągu: {'Włączony' if env.deploy_train_mode else 'Wyłączony'}")
            first_station_clicked_idx = None

    return action


if __name__ == "__main__":
    pygame.init()

    env = MiniMetroEnv(render_mode="human")
    obs, info = env.reset()

    print("--- MiniMetro Gym Environment ---")
    print("Sterowanie:")
    print("  Lewy klik: Wybierz stację 1, potem stację 2, aby zbudować linię")
    print("  Prawy klik: Usuń linię o aktualnie wybranym kolorze")
    print("  Klawisze 1-6: Wybierz aktywny kolor linii")
    print("  Spacja: Przełącz tryb rozmieszczania pociągu")
    print("  Kliknięcie w przycisk 'Train': Przełącza tryb rozmieszczania pociągu")
    print("  ESC: Wyjście z gry")

    running = True
    while running:
        action_to_take = create_noop_action()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            mapped_action = map_mouse_to_action(env, event)
            if mapped_action["high_level_action"] != 0:
                action_to_take = mapped_action

        obs, reward, terminated, truncated, info = env.step(action_to_take)
        env.render()

        if reward != 0:
            print(f"Otrzymano nagrodę: {reward:.2f}, Aktualny wynik: {info['score']}")

        if terminated or truncated:
            print("=" * 20)
            print(f"Gra zakończona! Wynik końcowy: {info['score']}, Tydzień: {info['week_number']}")
            print("=" * 20)
            pygame.time.wait(5000)
            running = False

    env.close()
    pygame.quit()
    sys.exit()