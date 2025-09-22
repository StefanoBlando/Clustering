"""
Business interpretation and strategic insights
Estratto da MODULO INTERPRETAZIONE BUSINESS MULTI-PROSPETTIVA
"""

import pandas as pd
import numpy as np

def generate_demographic_profile(df, labels, cluster_id):
    """
    Genera profilo demografico per cluster
    Estratto da business profiling functions
    """
    cluster_data = df[labels == cluster_id]
    n_obs = len(cluster_data)
    
    profile = {
        'cluster_id': cluster_id,
        'size': n_obs,
        'size_percentage': n_obs / len(df) * 100
    }
    
    # Età
    if 'q1' in cluster_data.columns:
        profile['age_mean'] = cluster_data['q1'].mean()
        profile['age_std'] = cluster_data['q1'].std()
    
    # Genere
    if 'q2' in cluster_data.columns:
        gender_dist = cluster_data['q2'].value_counts(normalize=True)
        if len(gender_dist) > 0:
            profile['dominant_gender'] = gender_dist.index[0]
            profile['gender_dominance_pct'] = gender_dist.iloc[0] * 100
    
    # Educazione
    if 'q3' in cluster_data.columns:
        edu_dist = cluster_data['q3'].value_counts(normalize=True)
        if len(edu_dist) > 0:
            profile['dominant_education'] = edu_dist.index[0]
            profile['education_dominance_pct'] = edu_dist.iloc[0] * 100
    
    # Occupazione
    if 'q4' in cluster_data.columns:
        occ_dist = cluster_data['q4'].value_counts(normalize=True)
        if len(occ_dist) > 0:
            profile['dominant_occupation'] = occ_dist.index[0]
            profile['occupation_dominance_pct'] = occ_dist.iloc[0] * 100
    
    # Reddito
    if 'q6' in cluster_data.columns:
        income_dist = cluster_data['q6'].value_counts(normalize=True)
        if len(income_dist) > 0:
            profile['dominant_income'] = income_dist.index[0]
            profile['income_dominance_pct'] = income_dist.iloc[0] * 100
    
    return profile

def generate_sustainability_profile(df, labels, cluster_id):
    """
    Genera profilo sostenibilità per cluster
    Estratto da sustainability profiling
    """
    cluster_data = df[labels == cluster_id]
    
    profile = {'cluster_id': cluster_id}
    
    # Atteggiamenti ambientali (Likert 1-7)
    environmental_vars = ['q7', 'q8', 'q9']  # Awareness, concern, urgency
    existing_env = [var for var in environmental_vars if var in cluster_data.columns]
    
    if existing_env:
        env_scores = [cluster_data[var].mean() for var in existing_env]
        profile['environmental_attitude'] = np.mean(env_scores)
        profile['attitude_classification'] = classify_attitude_level(profile['environmental_attitude'])
    
    # Comportamenti sostenibili
    behavior_vars = ['q13', 'q19', 'q20', 'q21']  # Waste, sorting, energy, water
    existing_beh = [var for var in behavior_vars if var in cluster_data.columns]
    
    if existing_beh:
        beh_scores = [cluster_data[var].mean() for var in existing_beh]
        profile['sustainable_behavior'] = np.mean(beh_scores)
        profile['behavior_classification'] = classify_behavior_level(profile['sustainable_behavior'])
    
    # Intenzioni future
    if 'q23' in cluster_data.columns:
        profile['future_intentions'] = cluster_data['q23'].mean()
    
    # Barriere percepite (se disponibili)
    if 'q25' in cluster_data.columns:
        barriers = cluster_data['q25'].value_counts()
        if len(barriers) > 0:
            profile['main_barrier'] = barriers.index[0]
            profile['barrier_percentage'] = barriers.iloc[0] / len(cluster_data) * 100
    
    return profile

def classify_attitude_level(score):
    """Classifica livello attitudini ambientali"""
    if score >= 6.0:
        return "Molto Favorevole"
    elif score >= 5.0:
        return "Favorevole"
    elif score >= 4.0:
        return "Moderato"
    elif score >= 3.0:
        return "Scettico"
    else:
        return "Molto Scettico"

def classify_behavior_level(score):
    """Classifica livello comportamenti sostenibili"""
    if score >= 6.0:
        return "Molto Attivo"
    elif score >= 5.0:
        return "Attivo"
    elif score >= 4.0:
        return "Moderato"
    elif score >= 3.0:
        return "Poco Attivo"
    else:
        return "Inattivo"

def generate_strategic_recommendations(demographic_profile, sustainability_profile):
    """
    Genera raccomandazioni strategiche
    Estratto da strategic insights generation
    """
    cluster_id = demographic_profile['cluster_id']
    size_pct = demographic_profile['size_percentage']
    
    recommendations = {
        'cluster_id': cluster_id,
        'priority_level': 'High' if size_pct > 30 else 'Medium' if size_pct > 20 else 'Low',
        'marketing_strategies': [],
        'communication_approaches': [],
        'product_positioning': [],
        'barriers_to_address': []
    }
    
    # Strategie basate su dimensione
    if size_pct > 30:
        recommendations['marketing_strategies'].append("Mass market approach")
        recommendations['marketing_strategies'].append("Mainstream media channels")
    elif size_pct > 20:
        recommendations['marketing_strategies'].append("Targeted campaigns")
        recommendations['marketing_strategies'].append("Niche media channels")
    else:
        recommendations['marketing_strategies'].append("Specialized targeting")
        recommendations['marketing_strategies'].append("Direct marketing")
    
    # Strategie basate su demografia
    if demographic_profile.get('dominant_occupation') == 'Studente/ Studentessa':
        recommendations['communication_approaches'].extend([
            "Social media focus",
            "Educational content",
            "Peer influence campaigns",
            "University partnerships"
        ])
        recommendations['product_positioning'].extend([
            "Affordable options",
            "Student discounts",
            "Future-oriented messaging"
        ])
    
    # Strategie basate su sostenibilità
    attitude_class = sustainability_profile.get('attitude_classification', '')
    if 'Favorevole' in attitude_class:
        recommendations['communication_approaches'].extend([
            "Reinforce existing beliefs",
            "Advanced sustainability topics",
            "Leadership positioning"
        ])
    elif 'Moderato' in attitude_class:
        recommendations['communication_approaches'].extend([
            "Education and awareness",
            "Practical benefits focus",
            "Step-by-step guidance"
        ])
    
    # Barriere da affrontare
    main_barrier = sustainability_profile.get('main_barrier', '')
    if 'cost' in main_barrier.lower():
        recommendations['barriers_to_address'].extend([
            "Cost-benefit demonstrations",
            "Financing options",
            "Long-term savings emphasis"
        ])
    
    return recommendations

def create_cluster_business_summary(df, labels, method_name):
    """
    Crea summary business completo
    Estratto da business summary generation
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    summary = {
        'method': method_name,
        'n_clusters': n_clusters,
        'clusters': [],
        'overall_insights': []
    }
    
    for cluster_id in unique_labels:
        # Profili
        demo_profile = generate_demographic_profile(df, labels, cluster_id)
        sust_profile = generate_sustainability_profile(df, labels, cluster_id)
        recommendations = generate_strategic_recommendations(demo_profile, sust_profile)
        
        # Etichetta interpretativa
        cluster_label = create_cluster_label(demo_profile, sust_profile)
        
        cluster_summary = {
            'cluster_id': cluster_id,
            'label': cluster_label,
            'size_percentage': demo_profile['size_percentage'],
            'demographic_profile': demo_profile,
            'sustainability_profile': sust_profile,
            'strategic_recommendations': recommendations
        }
        
        summary['clusters'].append(cluster_summary)
    
    # Insights generali
    summary['overall_insights'] = generate_overall_insights(summary['clusters'])
    
    return summary

def create_cluster_label(demo_profile, sust_profile):
    """Crea etichetta interpretativa per cluster"""
    
    # Elementi demografici chiave
    age = demo_profile.get('age_mean', 0)
    gender = demo_profile.get('dominant_gender', '')
    occupation = demo_profile.get('dominant_occupation', '')
    education = demo_profile.get('dominant_education', '')
    
    # Elementi sostenibilità
    attitude_class = sust_profile.get('attitude_classification', '')
    behavior_class = sust_profile.get('behavior_classification', '')
    
    # Logica per etichette
    if 'Studente' in occupation:
        if 'Favorevole' in attitude_class:
            return "Eco-Students (Future Leaders)"
        else:
            return "Young Learners (Emerging Awareness)"
    
    elif age > 45:
        if 'Attivo' in behavior_class:
            return "Mature Practitioners (Established Sustainable)"
        else:
            return "Traditional Consumers (Limited Engagement)"
    
    elif 'Laurea' in education:
        if 'Favorevole' in attitude_class and 'Attivo' in behavior_class:
            return "Educated Activists (High Commitment)"
        elif 'Favorevole' in attitude_class:
            return "Aware Professionals (Attitude-Action Gap)"
        else:
            return "Educated Moderates (Pragmatic Approach)"
    
    else:
        return f"{attitude_class} {behavior_class} Consumers"

def generate_overall_insights(clusters):
    """Genera insights generali cross-cluster"""
    
    insights = []
    
    # Analisi dimensioni
    sizes = [c['size_percentage'] for c in clusters]
    if max(sizes) > 40:
        insights.append("One dominant segment identified - focus mass market approach")
    elif len([s for s in sizes if s > 25]) >= 2:
        insights.append("Multiple major segments - multi-tier strategy needed")
    else:
        insights.append("Fragmented market - niche targeting approaches")
    
    # Analisi attitudini
    attitudes = [c['sustainability_profile'].get('attitude_classification', '') for c in clusters]
    favorable_count = sum(1 for a in attitudes if 'Favorevole' in a)
    
    if favorable_count == len(clusters):
        insights.append("Universal pro-environmental attitudes - focus on behavior facilitation")
    elif favorable_count > len(clusters) / 2:
        insights.append("Majority favorable attitudes - mixed approach needed")
    else:
        insights.append("Attitude heterogeneity - education and awareness priority")
    
    return insights
