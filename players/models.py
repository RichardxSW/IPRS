from django.db import models

# Create your models here.
class Test(models.Model):
    tes=models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # unique_together = ('tes')
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.tes}"
    

class Dataset(models.Model):
    league_name=models.CharField(max_length=50)
    season=models.CharField(max_length=50)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('league_name', 'season')
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.league_name} ({self.season})"
    

class Player(models.Model):
    dataset=models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='players')
    player=models.CharField(max_length=100)
    team=models.CharField(max_length=50, blank=True, null=True)
    nationality=models.CharField(max_length=50, blank=True, null=True)
    position=models.CharField(max_length=50, blank=True, null=True)
    age=models.PositiveIntegerField(default=0)
    appearance=models.PositiveIntegerField(default=0)
    total_minute=models.PositiveIntegerField(default=0)
    total_goal=models.PositiveIntegerField(default=0)
    goal_per_game=models.FloatField(default=0)
    shot_per_game=models.FloatField(default=0)
    sot_per_game=models.FloatField(default=0)
    assist=models.PositiveIntegerField(default=0)
    assist_per_game=models.FloatField(default=0)
    successful_dribble_per_game=models.FloatField(default=0)
    key_pass_per_game=models.FloatField(default=0)
    successful_pass_per_game=models.FloatField(default=0)
    long_ball_per_game=models.FloatField(default=0)
    successful_crossing_per_game=models.FloatField(default=0)
    ball_recovered_per_game=models.FloatField(default=0)
    dribbled_past_per_game=models.FloatField(default=0)
    clearance_per_game=models.FloatField(default=0)
    error=models.PositiveIntegerField(default=0)
    error_per_game=models.FloatField(default=0)
    total_duel_per_game=models.FloatField(default=0)
    aerial_duel_per_game=models.FloatField(default=0)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering=['-uploaded_at']

    def __str__(self):
        return f"{self.player}"
