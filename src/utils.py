def business_explanation(result):
    if result == 1:
        return (
            "Based on applicant income, credit history, and combined predictions "
            "from multiple machine learning models, the applicant is likely to "
            "repay the loan. Hence, the loan is approved."
        )
    else:
        return (
            "Based on applicant income, credit history, and combined predictions "
            "from multiple machine learning models, the applicant is unlikely to "
            "repay the loan. Hence, the loan is rejected."
        )
