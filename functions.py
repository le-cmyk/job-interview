from libraries import *


def reponse_question(df: pd.DataFrame, question: str, fonction: Callable[[pd.DataFrame], dict], commentaire: str = "") -> None:
    """
    Displays an expander in Streamlit to answer a specific question.

    Args:
    df (dataframe): The input pandas DataFrame
    question (str): The question to display in the expander.
    fonction (callable): The function that answers the question.
        The function should take a pandas DataFrame as its only argument and return a dictionary with keys as descriptions
        and values as the corresponding answers.
    commentaire (str, optional): Additional comments or explanations. Defaults to "".

    Returns:
    None
    """
    # Create an expander for the question
    q = st.expander(question)

    # Split the expander into two columns
    q_result, q_code = q.columns(2)

    # Display the source code of the provided function
    q_code.code(inspect.getsource(fonction))

    # If there is a comment, display it
    if commentaire != "":
        q_result.markdown("### Comments:\n" + commentaire)

    # Provide the answer by calling the provided function
    answers = fonction(df)
    
    # Display each element of the dictionary as "key: value"
    for key, value in answers.items():
        q_result.write(f"{key}:")
        if isinstance(value, plt.Figure):
            q_result.pyplot(value)
        else :
              
            q_result.write(value)