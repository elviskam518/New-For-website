
import numpy as np
import pandas as pd
from scipy.special import expit

np.random.seed(42)


def generate_tech_hiring_data(n=15000, output_csv="tech_diversity_hiring_data.csv"):

    print("=" * 70)
    print("Generating Tech Diversity Hiring Dataset")
    print("Role: Software Engineer")
    print("=" * 70)

    gender = np.random.choice(
        ["Male", "Female"],
        size=n,
        p=[0.70, 0.30]
    )

    race = np.random.choice(
        ["White", "Black", "Asian", "Hispanic"],
        size=n,
        p=[0.50, 0.15, 0.25, 0.10]
    )

    age = np.random.normal(32, 7, n)
    age = np.clip(age, 21, 58).astype(int)

    visa_status = np.zeros(n, dtype=int)
    for i in range(n):
        if race[i] == "Asian":
            visa_status[i] = np.random.choice([0, 1, 2, 3], p=[0.35, 0.15, 0.30, 0.20])
        elif race[i] == "White":
            visa_status[i] = np.random.choice([0, 1, 2, 3], p=[0.85, 0.08, 0.05, 0.02])
        elif race[i] == "Hispanic":
            visa_status[i] = np.random.choice([0, 1, 2, 3], p=[0.75, 0.10, 0.08, 0.07])
        else:
            visa_status[i] = np.random.choice([0, 1, 2, 3], p=[0.80, 0.10, 0.06, 0.04])

    visa_map = {0: "Citizen", 1: "Green Card", 2: "H1B", 3: "Needs Sponsorship"}

    education = np.random.choice(
        [1, 2, 3, 4],
        n,
        p=[0.45, 0.38, 0.07, 0.10]
    )
    edu_map = {1: "Bachelor", 2: "Master", 3: "PhD", 4: "Self-taught/Bootcamp"}

    university_tier = np.zeros(n, dtype=int)
    for i in range(n):
        if education[i] == 4: 
            university_tier[i] = 4
        else:
            if race[i] == "White":
                university_tier[i] = np.random.choice([1, 2, 3], p=[0.15, 0.40, 0.45])
            elif race[i] == "Asian":
                university_tier[i] = np.random.choice([1, 2, 3], p=[0.20, 0.42, 0.38])
            elif race[i] == "Black":
                university_tier[i] = np.random.choice([1, 2, 3], p=[0.08, 0.32, 0.60])
            else:
                university_tier[i] = np.random.choice([1, 2, 3], p=[0.07, 0.33, 0.60])

    cs_degree = np.random.choice([0, 1], n, p=[0.25, 0.75])

    # Years of experience
    experience = np.random.exponential(4.5, n)
    experience = np.clip(experience, 0, 20)

    github_score = np.random.beta(2, 5, n) * 100

    algorithm_skill = (
        40
        + 1.5 * experience
        + 8 * (education == 2)
        + 12 * (education == 3)
        + 5 * cs_degree
        + np.random.normal(0, 15, n)
    )
    algorithm_skill = np.clip(algorithm_skill, 0, 100)

    system_design_skill = (
        30
        + 3.0 * experience
        + 5 * (education >= 2)
        + np.random.normal(0, 12, n)
    )
    system_design_skill = np.clip(system_design_skill, 0, 100)

    num_languages = np.clip(
        np.random.poisson(3, n) + (experience > 5).astype(int),
        1, 10
    )
    past_company = np.zeros(n, dtype=int)
    for i in range(n):
        if experience[i] < 1:
            past_company[i] = 4
        else:
            base_prob = [0.10, 0.25, 0.55, 0.10]
            if race[i] == "Asian":
                base_prob = [0.15, 0.30, 0.45, 0.10]
            elif race[i] in ["Black", "Hispanic"]:
                base_prob = [0.05, 0.18, 0.62, 0.15]
            past_company[i] = np.random.choice([1, 2, 3, 4], p=base_prob)

    company_map = {1: "FAANG", 2: "Well-known Tech", 3: "Average Company", 4: "No Experience"}


    ethnic_name = np.zeros(n, dtype=int)
    ethnic_name[race == "Black"] = np.random.choice([0, 1], np.sum(race == "Black"), p=[0.35, 0.65])
    ethnic_name[race == "Hispanic"] = np.random.choice([0, 1], np.sum(race == "Hispanic"), p=[0.30, 0.70])
    ethnic_name[race == "Asian"] = np.random.choice([0, 1], np.sum(race == "Asian"), p=[0.40, 0.60])

    has_referral = np.zeros(n, dtype=int)
    for i in range(n):
        if gender[i] == "Male" and race[i] in ["White", "Asian"]:
            has_referral[i] = np.random.choice([0, 1], p=[0.55, 0.45])
        elif gender[i] == "Male":
            has_referral[i] = np.random.choice([0, 1], p=[0.70, 0.30])
        elif gender[i] == "Female" and race[i] in ["White", "Asian"]:
            has_referral[i] = np.random.choice([0, 1], p=[0.65, 0.35])
        else:
            has_referral[i] = np.random.choice([0, 1], p=[0.78, 0.22])

    linkedin_score = np.random.beta(5, 2, n) * 100

    interviewer_type = np.random.choice([0, 1, 2], n, p=[0.25, 0.50, 0.25])
    resume_base = (
        30
        + 2.5 * experience
        + 0.15 * algorithm_skill
        + 0.10 * system_design_skill
        + 8 * (education == 2)
        + 12 * (education == 3)
        + 5 * cs_degree
        + 10 * (university_tier == 1)
        + 5 * (university_tier == 2)
        - 5 * (university_tier == 4)
        + 12 * (past_company == 1)
        + 6 * (past_company == 2)
        + 8 * has_referral
        + np.random.normal(0, 8, n)
    )

    resume_bias = np.zeros(n)

    resume_bias += np.where(ethnic_name == 1, -6, 0)

    resume_bias += np.where(gender == "Female", -3, 0)

    resume_bias = resume_bias * (interviewer_type * 0.4 + 0.2)

    resume_score = np.clip(resume_base + resume_bias, 0, 100)


    tech_interview_base = (
        0.5 * algorithm_skill
        + 0.3 * system_design_skill
        + 0.1 * experience
        + np.random.normal(0, 10, n)
    )

    tech_interview_bias = np.zeros(n)

    tech_interview_bias += np.where(gender == "Female", -4, 0)

    tech_interview_bias += np.where(race == "Black", -5, 0)
    tech_interview_bias += np.where(race == "Hispanic", -4, 0)

    tech_interview_bias += np.where(race == "Asian", 2, 0)

    tech_interview_bias = tech_interview_bias * (interviewer_type * 0.35 + 0.15)

    tech_interview_score = np.clip(tech_interview_base + tech_interview_bias, 0, 100)


    culture_fit_base = (
        50
        + 0.5 * experience
        + 0.1 * linkedin_score
        + np.random.normal(0, 10, n)
    )

    culture_fit_bias = np.zeros(n)

    culture_fit_bias += np.where((gender == "Male") & (race == "White"), 8, 0)
    culture_fit_bias += np.where((gender == "Male") & (race == "Asian"), 4, 0)

    culture_fit_bias += np.where(gender == "Female", -6, 0)

    culture_fit_bias += np.where(race == "Black", -7, 0)
    culture_fit_bias += np.where(race == "Hispanic", -5, 0)

    culture_fit_bias += np.where((gender == "Female") & (race == "Black"), -5, 0)
    culture_fit_bias += np.where((gender == "Female") & (race == "Hispanic"), -3, 0)

    culture_fit_bias += np.where(age > 40, -5, 0)
    culture_fit_bias += np.where(age > 50, -8, 0)

    culture_fit_bias = culture_fit_bias * (interviewer_type * 0.5 + 0.1)

    culture_fit_score = np.clip(culture_fit_base + culture_fit_bias, 0, 100)


    overall_interview = (
        0.25 * resume_score
        + 0.35 * tech_interview_score
        + 0.40 * culture_fit_score
    )

    logit = (
        -4.5
        + 0.06 * overall_interview
        + 0.08 * experience
        + 0.25 * (education == 2)
        + 0.40 * (education == 3)
        + 0.50 * (past_company == 1)
        + 0.25 * (past_company == 2)
        + 0.35 * has_referral
    )

    hiring_bias = np.zeros(n)

    hiring_bias += np.where(gender == "Male", 0.15, 0)

    hiring_bias += np.where(race == "White", 0.12, 0)
    hiring_bias += np.where(race == "Asian", 0.05, 0)
    hiring_bias += np.where(race == "Black", -0.18, 0)
    hiring_bias += np.where(race == "Hispanic", -0.12, 0)

    hiring_bias += np.where(visa_status == 3, -0.30, 0)
    hiring_bias += np.where(visa_status == 2, -0.08, 0)

    hiring_bias += np.where((gender == "Female") & (race == "Black"), -0.20, 0)
    hiring_bias += np.where((gender == "Female") & (race == "Hispanic"), -0.12, 0)
    hiring_bias += np.where((gender != "Male") & (age > 40), -0.15, 0)

    hiring_bias += np.where(age > 45, -0.12, 0)
    hiring_bias += np.where(age > 50, -0.18, 0)

    noise = np.random.normal(0, 0.20, n)

    final_logit = logit + hiring_bias + noise
    prob = expit(final_logit)

    hired = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "Gender": gender,
        "Race": race,
        "Age": age,
        "VisaStatus": visa_status,
        "VisaLabel": [visa_map[v] for v in visa_status],

        "EducationLevel": education,
        "EducationLabel": [edu_map[e] for e in education],
        "UniversityTier": university_tier,
        "CS_Degree": cs_degree,

        "YearsExperience": np.round(experience, 1),
        "AlgorithmSkill": np.round(algorithm_skill, 1),
        "SystemDesignSkill": np.round(system_design_skill, 1),
        "NumLanguages": num_languages,
        "GitHubScore": np.round(github_score, 1),
        "PastCompanyTier": past_company,
        "PastCompanyLabel": [company_map[c] for c in past_company],

        "EthnicNameSignal": ethnic_name,
        "HasReferral": has_referral,
        "LinkedInScore": np.round(linkedin_score, 1),

        "InterviewerType": interviewer_type,
        "ResumeScore": np.round(resume_score, 1),
        "TechInterviewScore": np.round(tech_interview_score, 1),
        "CultureFitScore": np.round(culture_fit_score, 1),
        "OverallInterviewScore": np.round(overall_interview, 1),

        "HireProbability": np.round(prob, 4),
        "Hired": hired,

        "TotalBiasEffect": np.round(hiring_bias, 4)
    })

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n✓ Saved dataset: {output_csv}")
    print(f"✓ Samples: {n:,}")
    print("✓ Role: Software Engineer")

    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)
    print(f"Overall hire rate: {df['Hired'].mean():.2%}")
    print(f"Average experience: {df['YearsExperience'].mean():.1f} years")
    print(f"Average algorithm skill: {df['AlgorithmSkill'].mean():.1f}")

    print("\n" + "=" * 70)
    print("Gender Diversity Analysis")
    print("=" * 70)
    gender_table = df.groupby("Gender").agg({
        "Hired": ["mean", "sum", "count"],
        "AlgorithmSkill": "mean",
        "YearsExperience": "mean",
        "CultureFitScore": "mean"
    }).round(3)
    gender_table.columns = ["HireRate", "HiredCount", "ApplicantCount", "AvgAlgorithmSkill", "AvgExperience", "AvgCultureFit"]
    print(gender_table)

    print("\n" + "=" * 70)
    print("Race Diversity Analysis")
    print("=" * 70)
    race_table = df.groupby("Race").agg({
        "Hired": ["mean", "sum", "count"],
        "AlgorithmSkill": "mean",
        "YearsExperience": "mean",
        "CultureFitScore": "mean"
    }).round(3)
    race_table.columns = ["HireRate", "HiredCount", "ApplicantCount", "AvgAlgorithmSkill", "AvgExperience", "AvgCultureFit"]
    print(race_table)

    print("\n" + "=" * 70)
    print("Intersectional Analysis: Gender × Race")
    print("=" * 70)
    cross = df.pivot_table(
        values="Hired",
        index="Gender",
        columns="Race",
        aggfunc="mean"
    ).round(4)
    cross_pct = cross.applymap(lambda x: f"{x:.2%}")
    print(cross_pct)

    print("\n" + "=" * 70)
    print("Bias Check")
    print("=" * 70)

    high_skill = df[df["AlgorithmSkill"] >= df["AlgorithmSkill"].median()]

    print("\nWithin the high-skill group (AlgorithmSkill >= median):")
    print("-" * 50)

    male_white = high_skill[(high_skill["Gender"] == "Male") & (high_skill["Race"] == "White")]["Hired"].mean()
    female_black = high_skill[(high_skill["Gender"] == "Female") & (high_skill["Race"] == "Black")]["Hired"].mean()
    male_asian = high_skill[(high_skill["Gender"] == "Male") & (high_skill["Race"] == "Asian")]["Hired"].mean()
    female_white = high_skill[(high_skill["Gender"] == "Female") & (high_skill["Race"] == "White")]["Hired"].mean()

    print(f"White male hire rate: {male_white:.2%}")
    print(f"Asian male hire rate: {male_asian:.2%}")
    print(f"White female hire rate: {female_white:.2%}")
    print(f"Black female hire rate: {female_black:.2%}")
    print(f"\nGap (White male vs Black female): {male_white - female_black:.2%}")

    if male_white - female_black > 0.15:
        print("✓ Large gap remains even after controlling for skill → bias is substantial")

    print("\n" + "=" * 70)
    print("Age Bias Analysis")
    print("=" * 70)
    age_groups = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60], labels=["21-30", "31-40", "41-50", "51-60"])
    age_table = df.groupby(age_groups).agg({
        "Hired": "mean",
        "AlgorithmSkill": "mean",
        "YearsExperience": "mean"
    }).round(3)
    age_table.columns = ["HireRate", "AvgAlgorithmSkill", "AvgExperience"]
    print(age_table)

    print("\n" + "=" * 70)
    print("Interview Stage Bias Analysis")
    print("=" * 70)
    print("\nGender gaps by stage:")
    for stage in ["ResumeScore", "TechInterviewScore", "CultureFitScore"]:
        male_avg = df[df["Gender"] == "Male"][stage].mean()
        female_avg = df[df["Gender"] == "Female"][stage].mean()
        print(f"  {stage}: Male={male_avg:.1f}, Female={female_avg:.1f}, Gap={male_avg - female_avg:.1f}")

    print("\n" + "=" * 70)
    print("Dataset Feature Summary")
    print("=" * 70)
    print("• This dataset simulates multiple bias mechanisms in software engineer hiring")
    print("• Bias exists in: resume screening, technical interviews, culture fit, final decision")
    print("• Includes intersectional effects: gender×race, gender×age")
    print("• Useful for: bias detection testing, fair ML research, diversity analysis")

    return df


def generate_analysis_report(df, output_file="tech_hiring_analysis.txt"):

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Tech Hiring Diversity Analysis Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Overall Summary\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total applicants: {len(df):,}\n")
        f.write(f"Total hired: {df['Hired'].sum():,}\n")
        f.write(f"Overall hire rate: {df['Hired'].mean():.2%}\n\n")

        f.write("2. Gender Analysis\n")
        f.write("-" * 40 + "\n")
        for g in df["Gender"].unique():
            subset = df[df["Gender"] == g]
            f.write(f"{g}: Applicants={len(subset)}, HireRate={subset['Hired'].mean():.2%}\n")
        f.write("\n")

        f.write("3. Race Analysis\n")
        f.write("-" * 40 + "\n")
        for r in df["Race"].unique():
            subset = df[df["Race"] == r]
            f.write(f"{r}: Applicants={len(subset)}, HireRate={subset['Hired'].mean():.2%}\n")
        f.write("\n")

        f.write("4. Intersectional Analysis (Gender × Race)\n")
        f.write("-" * 40 + "\n")
        cross = df.groupby(["Gender", "Race"])["Hired"].mean()
        for (g, r), rate in cross.items():
            f.write(f"{g} + {r}: {rate:.2%}\n")

    print(f"✓ Analysis report saved: {output_file}")


if __name__ == "__main__":
    df = generate_tech_hiring_data(n=15000)

    generate_analysis_report(df)

    print("\n" + "=" * 70)
    print("Done! Generated files:")
    print("  1. tech_diversity_hiring_data.csv - main dataset")
    print("  2. tech_hiring_analysis.txt - analysis report")
    print("=" * 70)
