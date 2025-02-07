import string
import pandas as pd
from rapidfuzz import fuzz

def fuzzy_match(user_query, hr_role, hr_id):
    try:
        similar_jobs = []

        queries = user_query.lower().split()
        punctuation_to_remove = string.punctuation.replace('#', '').replace('+', '').replace('-', '')
        
        for job_profile, idx in zip(hr_role, hr_id):
            job_profile_token = job_profile.lower().split()
            
            # Remove unwanted punctuation from job profile tokens
            filtered_job_profile = [
                ''.join(char for char in word if char not in punctuation_to_remove)
                for word in job_profile_token
            ]
            filtered_job_profile = [word for word in filtered_job_profile if word]

            matches = [word for word in filtered_job_profile if max(fuzz.partial_ratio(word, query) for query in queries) > 80]

            if matches:
                job_title_keyword_ratio = max(
                    (fuzz.ratio(user_query.lower(), match) for match in matches), 
                    default=0
                )
                job_title = ' '.join(matches)
                job_title_ratio = fuzz.ratio(user_query.lower(), job_title)

                max_similarity = max(job_title_keyword_ratio, job_title_ratio)

                if max_similarity > 80:
                    # similar_jobs.append(idx)
                    similar_jobs.append(
                            {
                                "HR ID": str(idx),
                                "Job Title": str(job_profile)
                            }
                        )
        df_results = pd.DataFrame(similar_jobs)
        return df_results

    except Exception as e:
        print('➡ error in fuzzy match:', e)
        return []
