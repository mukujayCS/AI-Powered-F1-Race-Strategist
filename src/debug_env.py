"""Debug script to test environment rewards and behavior"""

from f1_env_final import create_f1_env

def test_no_pitstop_penalty():
    """Test that not pitting results in disqualification"""
    print("\n" + "="*60)
    print("TEST 1: No pit stops (should get -100 penalty)")
    print("="*60)

    env = create_f1_env()
    state, info = env.reset(options={"driver_id": 1})

    total_reward = 0
    done = False
    lap = 0

    # Never pit, just stay out
    while not done:
        state, reward, terminated, truncated, info = env.step(1)  # Action 1 = neutral pace, no pit
        done = terminated or truncated
        total_reward += reward
        lap += 1

        if lap % 10 == 0:
            print(f"  Lap {lap}: Reward={reward:.2f}, Total={total_reward:.2f}, "
                  f"Pos={info['position']}, Stops={info['num_pitstops']}")

    print(f"\nFinal: Total Reward={total_reward:.2f}, Position={info['position']}, Stops={info['num_pitstops']}")
    print(f"Expected: Massive negative reward (~-100 or worse)")
    print()

def test_one_pitstop():
    """Test that one pit stop avoids penalty"""
    print("\n" + "="*60)
    print("TEST 2: One pit stop at lap 25 (should be OK)")
    print("="*60)

    env = create_f1_env()
    state, info = env.reset(options={"driver_id": 1})

    total_reward = 0
    done = False
    lap = 0

    while not done:
        # Pit at lap 25
        if lap == 24:  # lap counter starts at 0
            action = 4  # Pit for medium tires
            print(f"  >>> PITTING at lap {lap+1}")
        else:
            action = 1  # Neutral pace

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        lap += 1

        if lap % 10 == 0 or lap == 25:
            print(f"  Lap {lap}: Reward={reward:.2f}, Total={total_reward:.2f}, "
                  f"Pos={info['position']}, Stops={info['num_pitstops']}")

    print(f"\nFinal: Total Reward={total_reward:.2f}, Position={info['position']}, Stops={info['num_pitstops']}")
    print(f"Expected: Should avoid -100 penalty and finish with reasonable position")
    print()

def test_too_many_pitstops():
    """Test that excessive pitting is penalized"""
    print("\n" + "="*60)
    print("TEST 3: Pit every 5 laps (should be penalized)")
    print("="*60)

    env = create_f1_env()
    state, info = env.reset(options={"driver_id": 1})

    total_reward = 0
    done = False
    lap = 0
    pit_count = 0

    while not done:
        # Pit every 5 laps
        if lap > 0 and lap % 5 == 0:
            action = 4  # Pit for medium tires
            pit_count += 1
        else:
            action = 1  # Neutral pace

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        lap += 1

        if lap % 10 == 0:
            print(f"  Lap {lap}: Reward={reward:.2f}, Total={total_reward:.2f}, "
                  f"Pos={info['position']}, Stops={info['num_pitstops']}")

    print(f"\nFinal: Total Reward={total_reward:.2f}, Position={info['position']}, Stops={info['num_pitstops']}")
    print(f"Expected: Negative reward due to excessive pits and position loss")
    print()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ENVIRONMENT REWARD DIAGNOSTICS")
    print("="*60)

    test_no_pitstop_penalty()
    test_one_pitstop()
    test_too_many_pitstops()

    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60)
