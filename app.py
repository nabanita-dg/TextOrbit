import streamlit as st

# Simple mood-to-emoji dictionary
mood_emojis = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😡",
    "excited": "🤩",
    "nervous": "😬",
    "bored": "😴",
}

def main():
    st.title("Mood Classifier 😎")
    st.subheader("Type your mood and get an emoji!")

    # User input
    user_mood = st.text_input("How are you feeling today?").lower()

    # Output
    if user_mood:
        emoji = mood_emojis.get(user_mood, "🤔")
        st.write(f"Your mood emoji is: {emoji}")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and Hugging Face Spaces.")

if __name__ == "__main__":
    main()