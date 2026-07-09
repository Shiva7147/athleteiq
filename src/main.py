# main.py
# AthleteIQ — Entry Point
# Run this file to start the AthleteIQ agent

from agent import build_agent, run_agent
from tools import athlete_data_log

def main():
    print("=" * 55)
    print("  AthleteIQ — AI Sports Science Agent")
    print("  Powered by LangChain + LangGraph + ChromaDB")
    print("=" * 55)

    print("\nInitialising agent...")
    agent = build_agent()
    print("Agent ready. Type your question or 'quit' to exit.\n")

    while True:
        user_input = input("Coach: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nAthlete data logged this session:")
            for entry in athlete_data_log:
                print(f"  {entry['timestamp']} | {entry['athlete']} "
                      f"| {entry['metric']} = {entry['value']}")
            print("\nAthleteIQ session ended. Stay safe out there.")
            break

        print("\nAthleteIQ: ", end="")
        response = run_agent(agent, user_input)
        print(response)
        print()

if __name__ == "__main__":
    main()
